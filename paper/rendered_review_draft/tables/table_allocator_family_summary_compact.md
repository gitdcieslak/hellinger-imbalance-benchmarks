| Model | AUROC | AP | R@0.50 | R@0.01 | Recovery | Persistence | Max Jump | Pattern |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CART | 0.625 | 0.146 | 0.291 | 0.291 | 0.000 | 0.291 | 0.000 | quantized_allocator |
| HDDT | 0.781 | 0.246 | 0.291 | 0.852 | 0.561 | 0.501 | 0.269 | broad_allocator |
| Bagged HDDT (hddt_forest) | 0.839 | 0.343 | 0.095 | 0.998 | 0.902 | 0.540 | 0.610 | cliff_allocator |
| Random Forest | 0.818 | 0.357 | 0.153 | 0.911 | 0.758 | 0.560 | 0.346 | broad_allocator |
| XGBoost | 0.801 | 0.309 | 0.121 | 0.985 | 0.865 | 0.491 | 0.489 | cliff_allocator |
| LightGBM | 0.848 | 0.346 | 0.203 | 0.751 | 0.548 | 0.441 | 0.222 | conservative_allocator |
| MLP | 0.707 | 0.241 | 0.135 | 0.787 | 0.652 | 0.487 | 0.406 | cliff_allocator |
