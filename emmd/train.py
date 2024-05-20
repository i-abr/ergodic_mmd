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
from emmd.score import ScoreDensity

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
def train_mmd_kernel_score(key, mmd_model, samples, to_train, epochs, opt=None, **kwargs):
    # parameters
    k = mmd_model.k
    R = kwargs.get("R", 100)
    z = deepcopy(jax.random.choice(key, samples, (R,), replace=False))
    mu = kwargs.get("mu", None)
    sigma = kwargs.get("sigma", None)
    model = ScoreDensity(key, deepcopy(k), z, mu=mu, sigma=sigma)
    params, static = trainable(model, to_train)

    #### data
    batch_size = kwargs.get("batch_size", R)
    shuffle = kwargs.get("shuffle", True)
    data_loader = jdl.DataLoader(
        jdl.ArrayDataset(samples), 'jax', 
        batch_size=batch_size, shuffle=shuffle, drop_last=True
    )

    #### optimizer
    if opt is None:
        lr = kwargs.get("lr", 1e-3)
        opt = optax.adamw(lr)
    opt_state = opt.init(params)

    #### define an opt step
    @jax.jit
    def opt_step(batch, _params, _opt_state):
        @jax.value_and_grad
        def loss_fn(_params):
            _model = eqx.combine(_params, static)
            return _model(batch)

        loss, grads = loss_fn(_params)
        updates, _opt_state = opt.update(grads, _opt_state , params=_params)
        _params = optax.apply_updates(_params, updates)
        return _params, _opt_state, loss

    # loop over epochs
    verbose = kwargs.get("verbose", False)
    print_iter = kwargs.get("print_iter", 50)
    loss_vals = []
    for epoch in range(epochs):
        batch_loss = []
        for batch in data_loader:
            params, opt_state, loss = opt_step(batch[0], params, opt_state)
            batch_loss.append(loss)

        loss_vals.append(sum(batch_loss) / len(batch_loss))
        loss_vals.append(loss)

        # # print output
        if verbose and epoch % print_iter == 0:
            print(f"epoch {epoch},loss: {loss}")

    model = eqx.combine(params, static)
    mmd_model = eqx.tree_at(lambda t: t.k, mmd_model, model.k)
    return mmd_model, model, jnp.array(loss_vals)


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

    # #### data
    # batch_size = kwargs.get("batch_size", None)
    # shuffle = kwargs.get("shuffle", True)
    # data = make_dataloader(key, samples, model.w.shape[0], batch_size=batch_size, shuffle=shuffle)
    # batches = jnp.array([batch[0] for batch in data])

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


def train_mmd_kernel_lgcp(
        key, model, centroids, counts, volumes, to_train, epochs, 
        opt=None, lowrank=False, n_samples=1, **kwargs
        ):
    
    # parameters
    model = deepcopy(model)
    diag = kwargs.get("diag", 1e-5)
    params, static = trainable(model, to_train)
    if lowrank:
        raise NotImplementedError("Low rank GP not implemented for LGCP.")
    else:
        nll_fn = lgcp_nll

    #### optimizer
    if opt is None:
        lr = kwargs.get("lr", 1e-3)
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr / 10, 
            peak_value=lr,
            warmup_steps= epochs // 10,
            decay_steps= epochs
        )
        opt = optax.adamw(learning_rate=schedule)
    opt_state = opt.init(params)

    #### define an opt step
    @jax.jit
    def opt_step(key, _params, _opt_state):
        @jax.value_and_grad
        def loss_fn(_params_):
            _model = eqx.combine(_params_, static)
            nll_loss = nll_fn(
                key, _model.k, centroids, counts, volumes, 
                diag=diag, n_samples=n_samples
            )

            return nll_loss

        loss, grads = loss_fn(_params)
        updates, _opt_state = opt.update(grads, _opt_state , params=_params)
        _params = optax.apply_updates(_params, updates)
        return _params, _opt_state, loss

    # loop over epochs
    verbose = kwargs.get("verbose", False)
    print_iter = kwargs.get("print_iter", 50)
    loss_vals = []
    for epoch in range(epochs):
        key, subkey = jax.random.split(key)
        params, opt_state, loss = opt_step(key, params, opt_state)
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
            mmd_val = _model(samples)
            return mmd_val + aux_loss(_model.w)

        loss, grads = loss_fn(_params)
        aux_loss_val = aux_loss(_params.w)
        updates, _opt_state = opt.update(grads, _opt_state , params=_params)
        _params = optax.apply_updates(_params, updates)
        return _params, _opt_state, [loss, loss - aux_loss_val, aux_loss_val]

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


# def train_mmd_grad_kernel(key, model, samples, to_train, epochs, opt=None, **kwargs):
#     # parameters
#     model = deepcopy(model)
#     params, static = trainable(model, to_train)

#     #### optimizer
#     if opt is None:
#         lr = kwargs.get("lr", 1e-3)
#         opt = optax.adamw(lr)
#     opt_state = opt.init(params)

#     #### define an opt step
#     @jax.jit
#     def opt_step(_params, _opt_state):
#         @jax.value_and_grad
#         def loss_fn(_params):
#             _model = eqx.combine(_params, static)

#             @jax.grad
#             def inner_loss(_particles):
#                 mmd_val = _model.two_sample_mmd(_particles, samples)
#                 return mmd_val

#             particle_grads = inner_loss(_model.w)
#             mmd_grad = jnp.mean(particle_grads**2)
#             return mmd_grad

#         loss, grads = loss_fn(_params)
#         updates, _opt_state = opt.update(grads, _opt_state , params=_params)
#         _params = optax.apply_updates(_params, updates)
#         return _params, _opt_state, loss

#     # loop over epochs
#     verbose = kwargs.get("verbose", False)
#     print_iter = kwargs.get("print_iter", 50)
#     loss_vals = []
#     for epoch in range(epochs):
#         params, opt_state, loss = opt_step(params, opt_state)
#         loss_vals.append(loss)

#         # # print output
#         if verbose and epoch % print_iter == 0:
#             print(f"epoch {epoch},loss: {loss}")

#     model = eqx.combine(params, static)
#     return model, jnp.array(loss_vals)



# def train_mmd_joint(key, model, samples, to_train, bounds, aux_loss=None, **kwargs):
#     #### parameters
#     _to_train = lambda t: [*to_train(t), t.w]
#     params, static = trainable(model, _to_train)

#     #### bounds
#     n_p = model.w.shape[0]
#     lb = eqx.tree_at(lambda t: t.w, params, jnp.tile(bounds[0], (n_p, 1)))
#     ub = eqx.tree_at(lambda t: t.w, params, jnp.tile(bounds[1], (n_p, 1)))
#     bounds = (lb, ub)
    
#     #### additional loss terms
#     if aux_loss is None:
#         aux_loss = lambda _samples: jnp.zeros(1)

#     #### define an opt step
#     @jax.jit
#     def loss_fn(params):
#         _model = eqx.combine(params, static)
#         mmd_val = _model(samples)
#         return mmd_val + aux_loss(_model.w)

#     #### optimizer
#     tol = kwargs.get("tol", 1e-4)
#     maxiter = kwargs.get("epochs", 10_000)
#     solver = jaxopt.LBFGSB(fun=loss_fn, maxiter=maxiter, tol=tol)
#     # res = solver.run(params, bounds=bounds)
#     params, state = solver.run(params, bounds=bounds)
#     model = eqx.combine(params, static)

#     return model, state

# # ------------------------------------- LOW RANK FIT ------------------------------------- #
# @jax.jit
# def dropout_lrgp(lrgp, key, sigma):
#     key, subkey = jax.random.split(key)
#     w = lrgp.kernel.kernel.w
#     dropout_mult = jnp.where(
#         sigma > 0., 
#         jnp.ones_like(w) + jax.random.normal(subkey, w.shape) * sigma, 
#         jnp.ones_like(w)
#     )
#     w_dropout = w * dropout_mult
#     lrgp = eqx.tree_at(lambda t: t.kernel.kernel.w, lrgp, w_dropout)
#     return lrgp


# def fit_lrgp(gp, y, epochs, to_train=None, dropout=0., **kwargs):
#     #### extract hyperparameters
#     lr = kwargs.pop("lr", 1e-3)
#     dropout_key = kwargs.pop("dropout_key", jax.random.PRNGKey(0))
#     dkeys = jax.random.split(dropout_key, epochs)

#     #### define trainable parameters
#     if to_train is not None:
#         param_fn = lambda t: trainable(t, to_train)
#     else:
#         param_fn = lambda t: freeze(t, lambda _t: _t.X)

#     #### define and initialize optimizer
#     # opt = optax.adamw(lr)
#     # params, static = param_fn(gp)
#     schedule = optax.warmup_cosine_decay_schedule(
#         init_value=0.0,
#         peak_value=lr,
#         warmup_steps=50,
#         decay_steps=epochs - epochs // 10,
#         end_value=kwargs.pop("lr_min", 5e-5),
#     )

#     opt = optax.adamw(learning_rate=schedule)
#     params, static = param_fn(gp)

#     #### define an opt step
#     @eqx.filter_jit
#     # def opt_step(params, dropout_params, opt_state):
#     def opt_step(params, opt_state, dkey):
#         @jax.value_and_grad
#         def loss_fn(_params):
#             model = eqx.combine(_params, static)
#             model = dropout_lrgp(model, dkey, dropout)
#             return model.nll(y)

#         # loss, grads = loss_fn(dropout_params)  # dropout_params loss
#         loss, grads = loss_fn(params)
#         updates, opt_state = opt.update(grads, opt_state, params=params)
#         params = optax.apply_updates(params, updates)
#         return params, opt_state, loss

#     #### loop over epochs
#     opt_state = opt.init(params)
#     verbose = kwargs.get("verbose", False)
#     print_iter = kwargs.get("print_iter", 50)
#     loss_vals = []
#     for epoch in range(epochs):
#         params, opt_state, loss = opt_step(params, opt_state, dkeys[epoch])
#         loss_vals.append(loss)

#         # # print output
#         if verbose and epoch % print_iter == 0:
#             print(f"epoch {epoch},loss: {loss}")

#     # return model
#     model = eqx.combine(params, static)
#     return model, jnp.array(loss_vals)