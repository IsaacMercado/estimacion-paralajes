# Validación con ArviZ

Este módulo no ajusta modelos. Sirve para evaluar trazas ya guardadas en formato
ArviZ/NetCDF después de correr los notebooks o Colab.

## Uso básico

```python
import arviz as az
import pandas as pd

from project.validation import (
    check_mcmc_diagnostics,
    evaluate_distance_posterior,
    summarize_elbo,
)

idata = az.from_netcdf("models/modelo_2_vi.nc")
sample = pd.read_parquet("data/processed/simulation_sample.parquet")

metrics = evaluate_distance_posterior(
    idata,
    true_distance_pc=sample["barycentric_distance"].to_numpy(),
    variable="distance_pc",
)
metrics
```

La condición indispensable es que `sample` esté en el mismo orden que la dimensión
`star` de la traza. Guarda siempre `source_id` junto con los resultados del modelo
para verificarlo.

## Métricas para simulaciones

`evaluate_distance_posterior` calcula:

- RMSE de la media posterior frente a la distancia verdadera.
- Sesgo medio y sesgo relativo medio.
- Cobertura empírica de intervalos 68% y 95%.
- Anchura media y mediana de intervalos.

Estas métricas son la prueba central para validar VI: si la cobertura 95% queda
muy por debajo de 0.95, la aproximación está subestimando incertidumbre.

## Diagnósticos MCMC

Para trazas NUTS:

```python
diagnostics = check_mcmc_diagnostics(idata, variable="distance_pc")
diagnostics.passed
diagnostics.messages
```

Umbrales usados por defecto:

- `R-hat < 1.01`.
- `ESS bulk > 400`.
- `ESS tail > 400`.
- cero divergencias.

No uses estos diagnósticos como prueba principal de VI; las muestras VI suelen
guardarse como una cadena artificial. Para VI usa cobertura en simulaciones,

## ELBO de VI

Después de correr NumPyro SVI:

```python
elbo_summary = summarize_elbo(vi_result.losses)
elbo_summary
```

Esto resume la pérdida inicial/final, mejor pérdida y estabilidad de la cola. Una
cola estable no prueba que VI sea correcto, pero una cola inestable indica que la
corrida no debe usarse para resultados finales.
