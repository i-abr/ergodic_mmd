# ---------------------------------------------------------------------------------------- #
#                                   GP TRAINING UTILITIES                                  #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import jit, vmap
import jax.tree_util as jtu
import optax
import jaxopt
from tensorflow_probability.substrates.jax import distributions as tfd
from copy import deepcopy
from sklearn.model_selection import KFold
import jax_dataloader as jdl
from copy import deepcopy

from emmd.gp import lrgp_nll, gp_nll, lgcp_nll

# -------------------------------------- PARAMETERS -------------------------------------- #
def freeze(model, frozen_fn):
    filter_spec = jtu.tree_map(lambda t: eqx.is_array(t), model)
    filter_spec = eqx.tree_at(frozen_fn, filter_spec, replace_fn=lambda _: False)
    return eqx.partition(model, filter_spec)


def trainable(model, trainable_prms):
    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(trainable_prms, filter_spec, replace_fn=lambda _: True)
    return eqx.partition(model, filter_spec)


# ------------------------------------- DATA LOADING ------------------------------------- #
def make_dataloader(key, samples, n_particles, batch_size=None, shuffle=True):
    n_s = samples.shape[0]

    if n_particles > n_s:
        raise ValueError(
            "Number of particles must be less than or equal to number of samples."
        )

    if shuffle:
        key, subkey = jax.random.split(key)
        samples = jax.random.permutation(subkey, samples, axis=0)

    if batch_size is None:
        batch_size = n_particles
    data_loader = jdl.DataLoader(
        jdl.ArrayDataset(samples), 'jax', batch_size=batch_size, shuffle=shuffle, drop_last=True
    )

    return data_loader


def permutation_dataloader(key, samples, n_perm=64, batch_size=16, shuffle=True):
    # permutations
    perm_keys = jax.random.split(key, n_perm)
    permutations = jax.vmap(
        lambda k: jax.random.permutation(k, samples, axis=0)
    )(perm_keys)

    # split data
    group1_size = samples.shape[0] // 2
    group2_size = group1_size - (samples.shape[0] % 2)
    group1 = permutations[:, :group1_size, :]
    group2 = permutations[:, group2_size:, :]

    data_loader = jdl.DataLoader(
        jdl.ArrayDataset(group1, group2), 'jax', batch_size=batch_size, shuffle=shuffle, drop_last=True
    )

    return data_loader


# ------------------------------------ KERNEL TRAINING ----------------------------------- #
def train_mmd_kernel(key, model, samples, to_train, epochs, opt=None, **kwargs):
    # parameters
    model = deepcopy(model)
    params, static = trainable(model, to_train)
    alpha = kwargs.get("alpha", 0)

    #### data
    batch_size = kwargs.get("batch_size", None)
    shuffle = kwargs.get("shuffle", True)
    data = make_dataloader(key, samples, model.w.shape[0], batch_size=batch_size, shuffle=shuffle)
    batches = jnp.array([batch[0] for batch in data])

    #### optimizer
    if opt is None:
        lr = kwargs.get("lr", 1e-3)
        opt = optax.adamw(lr)
    opt_state = opt.init(params)

    #### define an opt step
    @jax.jit
    def opt_step(_params, _opt_state):
        @jax.value_and_grad
        def loss_fn(_params):
            _model = eqx.combine(_params, static)
            power = vmap(
                lambda batch: _model.power(batch, alpha)
            )(batches)
            return jnp.mean(-power)

        loss, grads = loss_fn(_params)
        updates, _opt_state = opt.update(grads, _opt_state , params=_params)
        _params = optax.apply_updates(_params, updates)
        return _params, _opt_state, -loss

    # loop over epochs
    verbose = kwargs.get("verbose", False)
    print_iter = kwargs.get("print_iter", 50)
    loss_vals = []
    for epoch in range(epochs):
        params, opt_state, loss = opt_step(params, opt_state)
        loss_vals.append(loss)

        # # print output
        if verbose and epoch % print_iter == 0:
            print(f"epoch {epoch},loss: {loss}")

    model = eqx.combine(params, static)
    return model, jnp.array(loss_vals)


# ------------------------------------ SCORE MATCHING ------------------------------------ #
def train_mmd_kernel_score(key, model, samples, to_train, epochs, opt=None, **kwargs):
    # parameters
    train_alpha = kwargs.get("train_alpha", True)
    if train_alpha:
        def update_alpha(X, params, static):
            model = eqx.combine(params, static)
            new_alpha = model.compute_alpha(X)
            new_model = eqx.tree_at(lambda t: t.alpha, model, new_alpha)
            new_params, new_static = trainable(new_model, to_train)
            return new_params, new_static
    else:
        update_alpha = lambda X, params, not_static: params, static

    params, static = trainable(model, to_train)

    batch_size = kwargs.get("batch_size", 256)
    shuffle = kwargs.get("shuffle", True)
    data_loader = jdl.DataLoader(
        jdl.ArrayDataset(samples), 'jax', 
        batch_size=batch_size, shuffle=shuffle, drop_last=True
    )

    #### optimizer
    lr = kwargs.get("lr", 1e-3)
    if opt is None:
        # schedule = optax.warmup_cosine_decay_schedule(
        #     init_value=lr / 10,
        #     peak_value=lr,
        #     decay_steps=epochs,
        #     end_value=lr / 10**3,
        #     warmup_steps=epochs // 10
        # )
        # opt = optax.adamw(learning_rate=schedule)
        opt = optax.adamw(lr)
    opt_state = opt.init(params)

    #### define an opt step
    @jax.jit
    def opt_step(batch, _params, _static, _opt_state):    
        @jax.value_and_grad
        def loss_fn(_params_):
            _model = eqx.combine(_params_, static)
            return _model.score(batch)

        # regular old back prop
        loss, grads = loss_fn(_params)
        updates, _opt_state = opt.update(grads, _opt_state , params=_params)
        _params = optax.apply_updates(_params, updates)
        return _params, _static, _opt_state, loss

    # loop over epochs
    verbose = kwargs.get("verbose", False)
    print_iter = kwargs.get("print_iter", 50)
    loss_vals = []

    for epoch in range(epochs):
        batch_loss = []
        for batch in data_loader:
            # randomly split batch
            _batch = batch[0]
            key, subkey = jax.random.split(key)
            n_samples = _batch.shape[0] // 2
            b1_inds = jax.random.choice(subkey, jnp.arange(_batch.shape[0]), (n_samples,), replace=False)
            b2_inds = jnp.setdiff1d(jnp.arange(_batch.shape[0]), b1_inds)
            batch_alpha = _batch[b1_inds]
            batch_all = _batch[b2_inds]

            # update alpha
            params, static = update_alpha(batch_alpha, params, static)

            # take step
            params, static, opt_state, loss = opt_step(
                batch_all, params, static, opt_state
            )
            batch_loss.append(loss)

        loss_vals.append(sum(batch_loss) / len(batch_loss))
        loss_vals.append(loss)

        # # print output
        if verbose and epoch % print_iter == 0:
            print(f"epoch {epoch},loss: {loss}")

    params, static = update_alpha(samples, params, static)
    model = eqx.combine(params, static)

    return model, model, jnp.array(loss_vals)


# -------------------------------------- GP TRAINING ------------------------------------- #
def train_mmd_kernel_gp(key, model, X, y, to_train, epochs, opt=None, lowrank=False, **kwargs):
    # parameters
    model = deepcopy(model)
    diag = kwargs.get("diag", 1e-5)
    params, static = trainable(model, to_train)
    if lowrank:
        nll_fn = lrgp_nll
    else:
        nll_fn = gp_nll

    #### optimizer
    if opt is None:
        lr = kwargs.get("lr", 1e-3)
        opt = optax.adamw(lr)
    opt_state = opt.init(params)

    #### define an opt step
    @jax.jit
    def opt_step(_params, _opt_state):
        @jax.value_and_grad
        def loss_fn(_params_):
            _model = eqx.combine(_params_, static)
            return nll_fn(_model.k, X, y, diag)

        loss, grads = loss_fn(_params)
        updates, _opt_state = opt.update(grads, _opt_state , params=_params)
        _params = optax.apply_updates(_params, updates)
        return _params, _opt_state, loss

    # loop over epochs
    verbose = kwargs.get("verbose", False)
    print_iter = kwargs.get("print_iter", 50)
    loss_vals = []
    for epoch in range(epochs):
        params, opt_state, loss = opt_step(params, opt_state)
        loss_vals.append(loss)

        # # print output
        if verbose and epoch % print_iter == 0:
            print(f"epoch {epoch},loss: {loss}")

    model = eqx.combine(params, static)
    return model, jnp.array(loss_vals)




# ---------------------------------- TRAJECTORY TRAINING --------------------------------- #
def train_mmd(key, model, samples, bounds=None, aux_loss=None, optimizer="LBFGS", **kwargs):
    #### additional loss terms
    if aux_loss is None:
        print("No aux loss")
    if aux_loss is None:
        aux_loss = lambda _samples: 0.
    
    if optimizer == "LBFGS":
        return train_mmd_jaxopt(key, model, samples, bounds=bounds, aux_loss=aux_loss, **kwargs)

    elif optimizer == "optax":
        return train_mmd_optax(key, model, samples, aux_loss=aux_loss, **kwargs)
    
    else:
        raise ValueError("Unknown optimizer.")


def train_mmd_jaxopt(key, model, samples, bounds=None, aux_loss=None, **kwargs):
    #### parameters
    params, static = trainable(model, lambda t: t.w)

    #### optimizer
    opt_params = kwargs.get("opt_params", None)
    if opt_params is None:
        # opt_params = {"tol": 1e-4, "maxiter": 10_000}
        opt_params = {}
    print(opt_params)

    #### bounds
    if bounds is not None:
        n_p = model.w.shape[0]
        lb = eqx.tree_at(lambda t: t.w, params, jnp.tile(bounds[0], (n_p, 1)))
        ub = eqx.tree_at(lambda t: t.w, params, jnp.tile(bounds[1], (n_p, 1)))
        _bounds = (lb, ub)
    
    #### define an opt step
    @jax.jit
    def loss_fn(_params):
        _model = eqx.combine(_params, static)
        mmd_val = _model(samples)
        return mmd_val + aux_loss(_model.w)

    #### optimizer
    if bounds is not None:
        solver = jaxopt.LBFGSB(fun=loss_fn, **opt_params)
        params, state = solver.run(params, bounds=_bounds)
    else:
        solver = jaxopt.LBFGS(fun=loss_fn, **opt_params)
        params, state = solver.run(params)
    
    model = eqx.combine(params, static)

    return model, state


def train_mmd_optax(key, model, samples, epochs, aux_loss, **kwargs):
    #### parameters
    params, static = trainable(model, lambda t: t.w)

    #### optimizer
    opt = kwargs.get("opt", None)
    lr = kwargs.get("lr", 1e-2)
    if opt is None:
        schedule = kwargs.get("schedule", None)
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr / 10,
            peak_value=lr,
            decay_steps=epochs,
            end_value=lr / 10**3,
            warmup_steps=epochs // 10
        )
        opt = optax.adamw(learning_rate=schedule)
    opt_state = opt.init(params)

    #### define an opt step
    @jax.jit
    def opt_step(_params, _opt_state):
        @jax.value_and_grad
        def loss_fn(_params_):
            _model = eqx.combine(_params_, static)
            mmd_val = _model(samples)
            return mmd_val + aux_loss(_model.w)

        loss, grads = loss_fn(_params)
        aux_loss_val = aux_loss(_params.w)
        updates, _opt_state = opt.update(grads, _opt_state , params=_params)
        _params = optax.apply_updates(_params, updates)
        return _params, _opt_state, [loss, loss - aux_loss_val, aux_loss_val]

    # loop over epochs
    verbose = kwargs.get("verbose", False)
    print_iter = kwargs.get("print_iter", 100)
    loss_vals = []
    # return opt_step(params, opt_state)
    for epoch in range(epochs):
        params, opt_state, loss = opt_step(params, opt_state)
        loss_vals.append(loss)

        # # print output
        if verbose and epoch % print_iter == 0:
            print(f"epoch {epoch}, mmd loss: {loss[1]}, aux loss: {loss[2]}")

    model = eqx.combine(params, static)
    return model, jnp.array(loss_vals)


# -------------------------------------- DEPRECATED -------------------------------------- #

# def train_mmd_kernel_permutation(key, model, samples, to_train, epochs, opt=None, **kwargs):
#     # parameters
#     model = deepcopy(model)
#     params, static = trainable(model, to_train)
#     alpha = kwargs.get("alpha", 0)

#     #### data
#     n_perm = kwargs.get("n_perm", 64)
#     batch_size = kwargs.get("batch_size", 16)
#     shuffle = kwargs.get("shuffle", True)
#     batches = permutation_dataloader(
#         key, samples, n_perm=n_perm, batch_size=batch_size, shuffle=shuffle
#     )

#     #### optimizer
#     if opt is None:
#         lr = kwargs.get("lr", 1e-3)
#         opt = optax.adamw(lr)
#     opt_state = opt.init(params)

#     #### define an opt step
#     @jax.jit
#     def opt_step(_batch, _params, _opt_state):
#         @jax.value_and_grad
#         def loss_fn(_params):
#             _model = eqx.combine(_params, static)
#             mmd_val = vmap(
#                 lambda _x, _y: _model.two_sample_mmd(_x, _y)
#             )(_batch[0], _batch[1])
#             return jnp.mean(mmd_val)
#             # power = vmap(
#             #     lambda _x, _y: _model.two_sample_power(_x, _y, alpha)
#             # )(_batch[0], _batch[1])
#             # return jnp.mean(-power)

#         loss, grads = loss_fn(_params)
#         # return loss, grads
#         updates, _opt_state = opt.update(grads, _opt_state , params=_params)
#         _params = optax.apply_updates(_params, updates)
#         # return _params, _opt_state, -loss
#         return _params, _opt_state, loss

#     # loop over epochs
#     verbose = kwargs.get("verbose", False)
#     print_iter = kwargs.get("print_iter", 50)
#     loss_vals = []
#     for epoch in range(epochs):
#         losses = []
#         for batch in batches:
#             # return opt_step(batch, params, opt_state)
#             params, opt_state, loss = opt_step(batch, params, opt_state)
#             losses.append(loss)
        
#         loss_vals.append(sum(losses) / len(losses))

#         # # print output
#         if verbose and epoch % print_iter == 0:
#             print(f"epoch {epoch},loss: {loss}")

#     model = eqx.combine(params, static)
#     return model, jnp.array(loss_vals)
