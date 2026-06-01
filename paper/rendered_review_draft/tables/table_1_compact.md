| Model | AUROC | AP | R@0.50 | R@0.01 | Recovery | Smooth. | Max Jump | Pattern |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mlp_bce | 0.707 | 0.241 | 0.135 | 0.787 | 0.652 | 0.418 | 0.406 | cliff_allocator |
| mlp_oversampled | 0.835 | 0.299 | 0.497 | 0.704 | 0.207 | 0.539 | 0.068 | smooth_allocator |
| mlp_weighted | 0.773 | 0.266 | 0.579 | 0.851 | 0.272 | 0.608 | 0.178 | smooth_allocator |
| xgboost | 0.801 | 0.309 | 0.121 | 0.985 | 0.865 | 0.175 | 0.489 | cliff_allocator |
| hddt | 0.781 | 0.246 | 0.291 | 0.852 | 0.561 | 0.235 | 0.269 | broad_allocator |
