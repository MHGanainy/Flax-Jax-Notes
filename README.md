# JAX Notes and Cheat Sheet

## Warming up with JAX

### Key Features
- **Syntax Similarity**: JAX's syntax is similar to NumPy's.
- **SciPy API**: JAX supports a SciPy API (`jax.scipy`).
- **Special Transform Functions**: JAX provides unique transform functions.
- **Low-Level API**: JAX includes a low-level API called `lax`.

### Facts
1. **Similarity to NumPy**: JAX syntax is remarkably similar to NumPy.
2. **Immutable Arrays**: Unlike NumPy, JAX arrays are immutable.
   - Solution: Use `y = x.at[index].set(value)`
3. **Random Numbers Handling**: JAX handles random numbers differently and requires explicit PRNG state management.
   - Example: 
     ```python
     seed = 0
     key = random.PRNGKey(seed)
     x = random.normal(key, (10,))
     ```
4. **AI Accelerator Agnostic**: JAX code runs on various accelerators without modification.
   - Example:
     ```python
     x_jnp = random.normal(key, (size, size), dtype=jnp.float32)
     x_np = np.random.normal(size=(size, size)).astype(np.float32)
     ```

## JAX Transform Functions

### `jit`
- **Functionality**: Compiles functions using XLA for speed.
- **Benchmarking**:
  ```python
  data = random.normal(key, (1000000,))
  %timeit selu(data).block_until_ready()
  %timeit selu_jit(data).block_until_ready()
  ```

### `grad`
- **Automatic Differentiation**:
  ```python
  def sum_logistic(x):
      return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))
  
  x = jnp.arange(3.)
  grad_loss = grad(sum_logistic)
  print(grad_loss(x))
  ```

### `vmap`
- **Auto-vectorization**:
  ```python
  @jit
  def vmap_batched_apply_matrix(batched_x):
      return vmap(apply_matrix)(batched_x)
  ```

## JAX API Structure
- **NumPy <-> lax <-> XLA**: `lax` API is stricter and more powerful.
  - **Example**:
    ```python
    x = jnp.array([1, 2, 1])
    y = jnp.ones(10)
    result2 = lax.conv_general_dilated(
        x.reshape(1, 1, 3).astype(float),
        y.reshape(1, 1, 10),
        window_strides=(1,),
        padding=[(len(y) - 1, len(y) - 1)]
    )
    ```

## JIT in Depth

### How JIT Works
- **Compilation for Speed**:
  ```python
  def norm(X):
      X = X - X.mean(0)
      return X / X.std(0)

  norm_compiled = jit(norm)
  ```

### Handling Failures
- **Static Arguments**:
  ```python
  @partial(jit, static_argnums=(1,))
  def f(x, neg):
      return -x if neg else x
  ```

## Gotchas and Tips

### Pure Functions
- **Avoid Side Effects**:
  ```python
  def impure_print_side_effect(x):
      print("Executing function")
      return x
  ```

### In-Place Updates
- **Immutable Arrays**:
  ```python
  jax_array = jnp.zeros((3,3), dtype=jnp.float32)
  updated_array = jax_array.at[1, :].set(1.0)
  ```

### Out-of-Bounds Indexing
- **Non-Error Behavior**:
  ```python
  print(jnp.arange(10).at[11].add(23))
  print(jnp.arange(10)[11])
  ```

### Non-Array Inputs
- **Explicit Conversion**:
  ```python
  def permissive_sum(x):
      return jnp.sum(jnp.array(x))
  ```

### Random Numbers
- **PRNG State Management**:
  ```python
  key = random.PRNGKey(seed)
  key, subkey = random.split(key)
  ```

### Control Flow
- **Handling Control Flow with JIT**:
  ```python
  f_jit = jit(f, static_argnums=(0,))
  ```

### Debugging NaNs
- **NaN Debugging**:
  ```python
  from jax.config import config
  config.update("jax_debug_nans", True)
  ```

This cheat sheet provides a quick reference to the essential aspects and functionalities of JAX, including its syntax, key features, transform functions, API structure, and common pitfalls with solutions.
