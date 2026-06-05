# Final Exam Report Draft

## 1. Project goal

The project studies EEG-based prediction of cognitive and affective Performance Metrics using EEG/POW feature windows and deep learning models with local temporal context.

The main PM targets are:

- `target_excitement`
- `target_relaxation`
- `target_engagement`
- `target_interest`
- `target_focus`
- `target_stress`
- `target_attention`

## 2. Compared models

The current version compares the following model families:

- Classical ML baseline: tree-based and linear tabular models.
- MLP: multi-output neural baseline on single-window EEG/POW features.
- LSTM: sequence model using local temporal context.
- TransformerEncoder: attention-based sequence model using local temporal context.
- Autoencoder: reconstruction and synthetic generation of EEG/POW feature vectors.

## 3. Run directories

- Classical baseline: `20260604_163602_multi_pm_groupkfold_full_pow_plus_eeg_raw_pow`
- MLP: `20260605_130703_mlp_pm_groupkfold_full_pow_plus_eeg`
- LSTM: `20260605_131912_lstm_pm_groupkfold_full_pow_plus_eeg_seq5_center`
- TransformerEncoder: `20260605_133044_transformer_pm_groupkfold_full_pow_plus_eeg_seq5_center`
- Autoencoder: `20260605_133745_ae_pow_plus_eeg_full_pow_plus_eeg_latent32`

## 4. Main model comparison

| model_family       | target            |   mae_mean |   rmse_mean |     r2_mean |   pearson_mean |   spearman_mean |
|:-------------------|:------------------|-----------:|------------:|------------:|---------------:|----------------:|
| LSTM               | target_excitement |  0.133891  |   0.177177  |  0.371572   |       0.639532 |        0.59566  |
| MLP                | target_excitement |  0.146598  |   0.191453  |  0.266737   |       0.519571 |        0.493298 |
| TransformerEncoder | target_excitement |  0.124946  |   0.166256  |  0.435636   |       0.684282 |        0.62434  |
| LSTM               | target_relaxation |  0.112917  |   0.14181   |  0.260346   |       0.562732 |        0.522464 |
| MLP                | target_relaxation |  0.124115  |   0.189925  | -0.485615   |       0.3953   |        0.431151 |
| TransformerEncoder | target_relaxation |  0.106013  |   0.134988  |  0.329365   |       0.617705 |        0.580842 |
| LSTM               | target_engagement |  0.0928631 |   0.118472  |  0.128741   |       0.448996 |        0.368499 |
| MLP                | target_engagement |  0.0950306 |   0.139549  | -0.22738    |       0.370225 |        0.352387 |
| TransformerEncoder | target_engagement |  0.0954786 |   0.121465  |  0.078029   |       0.506554 |        0.448832 |
| LSTM               | target_interest   |  0.0644765 |   0.0866082 |  0.168258   |       0.460061 |        0.369615 |
| MLP                | target_interest   |  0.0668373 |   0.114191  | -0.862305   |       0.389981 |        0.303741 |
| TransformerEncoder | target_interest   |  0.0613443 |   0.0844023 |  0.214179   |       0.517787 |        0.429093 |
| LSTM               | target_focus      |  0.0867943 |   0.112441  |  0.167697   |       0.478776 |        0.446073 |
| MLP                | target_focus      |  0.0941458 |   0.131088  | -0.160469   |       0.295873 |        0.290149 |
| TransformerEncoder | target_focus      |  0.0862681 |   0.111848  |  0.170508   |       0.506852 |        0.472063 |
| LSTM               | target_stress     |  0.0904901 |   0.128906  |  0.137148   |       0.468422 |        0.414547 |
| MLP                | target_stress     |  0.0923718 |   0.13828   | -0.00536945 |       0.347865 |        0.355773 |
| TransformerEncoder | target_stress     |  0.088632  |   0.12783   |  0.148657   |       0.52246  |        0.450752 |
| LSTM               | target_attention  |  0.0963656 |   0.123679  |  0.0483079  |       0.379434 |        0.374309 |
| MLP                | target_attention  |  0.0946393 |   0.123504  |  0.0479648  |       0.327843 |        0.335147 |
| TransformerEncoder | target_attention  |  0.0957559 |   0.122702  |  0.0532111  |       0.444823 |        0.435877 |
| Classical baseline | nan               |  0.0968125 |   0.123563  |  0.185358   |       0.448678 |        0.44089  |
| Classical baseline | nan               |  0.0964981 |   0.123096  |  0.19151    |       0.460402 |        0.450927 |
| Classical baseline | nan               |  0.0973974 |   0.124637  |  0.171133   |       0.426413 |        0.412601 |
| Classical baseline | nan               |  0.0981885 |   0.123384  |  0.219805   |       0.479381 |        0.374119 |
| Classical baseline | nan               |  0.0978964 |   0.122989  |  0.224792   |       0.483666 |        0.377463 |
| Classical baseline | nan               |  0.0996058 |   0.12543   |  0.193715   |       0.449788 |        0.343722 |
| Classical baseline | nan               |  0.14352   |   0.182015  |  0.479184   |       0.708026 |        0.65733  |
| Classical baseline | nan               |  0.143374  |   0.181673  |  0.481138   |       0.71207  |        0.661239 |
| Classical baseline | nan               |  0.151163  |   0.1863    |  0.45437    |       0.705558 |        0.643026 |
| Classical baseline | nan               |  0.0948824 |   0.119292  |  0.175903   |       0.4515   |        0.40923  |
| Classical baseline | nan               |  0.0936603 |   0.118152  |  0.191577   |       0.463685 |        0.424515 |
| Classical baseline | nan               |  0.0910306 |   0.116903  |  0.208586   |       0.506242 |        0.475    |
| Classical baseline | nan               |  0.0545556 |   0.0753818 |  0.274711   |       0.55104  |        0.387841 |
| Classical baseline | nan               |  0.0547636 |   0.0753094 |  0.276104   |       0.55317  |        0.386397 |
| Classical baseline | nan               |  0.0511838 |   0.0736934 |  0.306836   |       0.556302 |        0.395262 |
| Classical baseline | nan               |  0.116955  |   0.148647  |  0.22136    |       0.504244 |        0.463732 |
| Classical baseline | nan               |  0.117205  |   0.148898  |  0.218721   |       0.505606 |        0.465547 |
| Classical baseline | nan               |  0.112189  |   0.144426  |  0.264956   |       0.52525  |        0.484072 |
| Classical baseline | nan               |  0.099532  |   0.138241  |  0.234601   |       0.486639 |        0.350543 |
| Classical baseline | nan               |  0.100107  |   0.139223  |  0.223692   |       0.476886 |        0.339394 |
| Classical baseline | nan               |  0.0973093 |   0.135271  |  0.267138   |       0.522947 |        0.416585 |
| Classical baseline | nan               |  0.0860579 |   0.110917  |  0.110645   |       0.458973 |        0.458933 |
| Classical baseline | nan               |  0.0869609 |   0.111545  |  0.100546   |       0.471315 |        0.471655 |
| Classical baseline | nan               |  0.0857562 |   0.111025  |  0.108909   |       0.431129 |        0.435084 |
| Classical baseline | nan               |  0.0820844 |   0.105499  |  0.305602   |       0.561812 |        0.47654  |
| Classical baseline | nan               |  0.0827818 |   0.106548  |  0.291732   |       0.556292 |        0.470746 |
| Classical baseline | nan               |  0.0793602 |   0.103606  |  0.3303     |       0.577329 |        0.491273 |
| Classical baseline | nan               |  0.158228  |   0.210992  |  0.190869   |       0.471377 |        0.408004 |
| Classical baseline | nan               |  0.157122  |   0.20935   |  0.203417   |       0.481461 |        0.416655 |
| Classical baseline | nan               |  0.152552  |   0.201763  |  0.260102   |       0.511526 |        0.429706 |
| Classical baseline | nan               |  0.087584  |   0.113961  |  0.224998   |       0.499655 |        0.500756 |
| Classical baseline | nan               |  0.0879254 |   0.11428   |  0.220644   |       0.490814 |        0.493698 |
| Classical baseline | nan               |  0.0887538 |   0.114938  |  0.211647   |       0.477833 |        0.473643 |
| Classical baseline | nan               |  0.0578389 |   0.0789677 |  0.117101   |       0.404778 |        0.351709 |
| Classical baseline | nan               |  0.0583184 |   0.0790532 |  0.115189   |       0.405725 |        0.352423 |
| Classical baseline | nan               |  0.0589034 |   0.0798643 |  0.0969385  |       0.36743  |        0.308568 |
| Classical baseline | nan               |  0.106136  |   0.133532  |  0.189972   |       0.449406 |        0.414433 |
| Classical baseline | nan               |  0.105693  |   0.133308  |  0.19269    |       0.457142 |        0.420903 |
| Classical baseline | nan               |  0.107356  |   0.134159  |  0.182352   |       0.430751 |        0.379226 |
| Classical baseline | nan               |  0.0717657 |   0.105233  |  0.1821     |       0.445148 |        0.37829  |
| Classical baseline | nan               |  0.072587  |   0.104819  |  0.188521   |       0.447082 |        0.369899 |
| Classical baseline | nan               |  0.0735762 |   0.10676   |  0.158183   |       0.410572 |        0.326654 |
| Classical baseline | nan               |  0.101705  |   0.128156  |  0.129469   |       0.400999 |        0.400033 |
| Classical baseline | nan               |  0.101035  |   0.127582  |  0.137238   |       0.414397 |        0.412405 |
| Classical baseline | nan               |  0.0986869 |   0.126118  |  0.156926   |       0.415668 |        0.414796 |
| Classical baseline | nan               |  0.092002  |   0.115623  |  0.200732   |       0.519788 |        0.42927  |
| Classical baseline | nan               |  0.0921644 |   0.116105  |  0.194063   |       0.521988 |        0.43336  |
| Classical baseline | nan               |  0.0895707 |   0.112624  |  0.241665   |       0.516681 |        0.416166 |
| Classical baseline | nan               |  0.143996  |   0.188668  |  0.404901   |       0.636379 |        0.579315 |
| Classical baseline | nan               |  0.143796  |   0.187836  |  0.410135   |       0.640738 |        0.579624 |
| Classical baseline | nan               |  0.144701  |   0.188853  |  0.403732   |       0.640458 |        0.575899 |
| Classical baseline | nan               |  0.100196  |   0.128092  |  0.0801275  |       0.333026 |        0.331052 |
| Classical baseline | nan               |  0.10039   |   0.12878   |  0.0702238  |       0.323619 |        0.327411 |
| Classical baseline | nan               |  0.0991643 |   0.12673   |  0.0995865  |       0.331129 |        0.323912 |
| Classical baseline | nan               |  0.0659762 |   0.091411  |  0.191856   |       0.447388 |        0.340884 |
| Classical baseline | nan               |  0.065532  |   0.0907898 |  0.202802   |       0.458731 |        0.357894 |
| Classical baseline | nan               |  0.0678591 |   0.0938401 |  0.148335   |       0.409948 |        0.309037 |
| Classical baseline | nan               |  0.133556  |   0.16201   |  0.200372   |       0.486411 |        0.468779 |
| Classical baseline | nan               |  0.135189  |   0.164076  |  0.179846   |       0.474312 |        0.461552 |
| Classical baseline | nan               |  0.13881   |   0.166493  |  0.1555     |       0.438268 |        0.424941 |
| Classical baseline | nan               |  0.11124   |   0.153456  |  0.158945   |       0.404994 |        0.330596 |
| Classical baseline | nan               |  0.11116   |   0.153792  |  0.155256   |       0.401472 |        0.339532 |
| Classical baseline | nan               |  0.112213  |   0.150414  |  0.191953   |       0.440973 |        0.313149 |
| Classical baseline | nan               |  0.0866918 |   0.113     |  0.171995   |       0.419589 |        0.385613 |
| Classical baseline | nan               |  0.086484  |   0.11288   |  0.173748   |       0.424146 |        0.397867 |
| Classical baseline | nan               |  0.0817726 |   0.109432  |  0.223454   |       0.478313 |        0.434261 |
| Classical baseline | nan               |  0.0846515 |   0.107053  |  0.356218   |       0.600373 |        0.526473 |
| Classical baseline | nan               |  0.0841559 |   0.106801  |  0.359246   |       0.603159 |        0.532168 |
| Classical baseline | nan               |  0.0797814 |   0.102642  |  0.408179   |       0.642481 |        0.565495 |
| Classical baseline | nan               |  0.147782  |   0.193914  |  0.287015   |       0.55473  |        0.518836 |
| Classical baseline | nan               |  0.148754  |   0.194625  |  0.281774   |       0.552542 |        0.508441 |
| Classical baseline | nan               |  0.153465  |   0.199279  |  0.247015   |       0.516613 |        0.464572 |
| Classical baseline | nan               |  0.0791766 |   0.101192  |  0.175866   |       0.421871 |        0.398663 |
| Classical baseline | nan               |  0.0787327 |   0.100717  |  0.18359    |       0.430962 |        0.405492 |
| Classical baseline | nan               |  0.0803882 |   0.102568  |  0.153304   |       0.394104 |        0.371934 |
| Classical baseline | nan               |  0.0684739 |   0.0907475 |  0.180889   |       0.435413 |        0.392785 |
| Classical baseline | nan               |  0.068773  |   0.091163  |  0.173372   |       0.427866 |        0.399287 |
| Classical baseline | nan               |  0.0691497 |   0.0903978 |  0.18719    |       0.45234  |        0.42936  |
| Classical baseline | nan               |  0.107835  |   0.134423  |  0.261631   |       0.531643 |        0.523209 |
| Classical baseline | nan               |  0.10729   |   0.133925  |  0.267084   |       0.534334 |        0.526735 |
| Classical baseline | nan               |  0.111203  |   0.136152  |  0.242512   |       0.537    |        0.5312   |
| Classical baseline | nan               |  0.0614984 |   0.0879495 |  0.212941   |       0.494673 |        0.351169 |
| Classical baseline | nan               |  0.0612878 |   0.0873701 |  0.223277   |       0.506794 |        0.386067 |
| Classical baseline | nan               |  0.0633629 |   0.0893649 |  0.187405   |       0.47595  |        0.327081 |
| Classical baseline | nan               |  0.092411  |   0.117779  |  0.0172707  |       0.322597 |        0.319087 |
| Classical baseline | nan               |  0.093105  |   0.11868   |  0.00218177 |       0.321534 |        0.316363 |
| Classical baseline | nan               |  0.0946286 |   0.120094  | -0.0217471  |       0.254071 |        0.250567 |
| Classical baseline | nan               |  0.0853521 |   0.11076   |  0.254386   |       0.567224 |        0.501384 |
| Classical baseline | nan               |  0.0852987 |   0.11087   |  0.25291    |       0.565866 |        0.509236 |
| Classical baseline | nan               |  0.084584  |   0.10959   |  0.270056   |       0.551974 |        0.467254 |
| Classical baseline | nan               |  0.12262   |   0.15903   |  0.40961    |       0.650287 |        0.556587 |
| Classical baseline | nan               |  0.122681  |   0.159572  |  0.40558    |       0.644639 |        0.547797 |
| Classical baseline | nan               |  0.134154  |   0.169421  |  0.329945   |       0.611862 |        0.537136 |
| Classical baseline | nan               |  0.0888926 |   0.114355  |  0.0235616  |       0.301409 |        0.25865  |
| Classical baseline | nan               |  0.0893841 |   0.115537  |  0.00325835 |       0.288357 |        0.251727 |
| Classical baseline | nan               |  0.0841495 |   0.112353  |  0.0574371  |       0.344172 |        0.316948 |
| Classical baseline | nan               |  0.0737261 |   0.103579  |  0.12613    |       0.420057 |        0.29729  |
| Classical baseline | nan               |  0.0750705 |   0.105543  |  0.0926711  |       0.405368 |        0.284267 |
| Classical baseline | nan               |  0.0702555 |   0.10236   |  0.146578   |       0.430944 |        0.330192 |
| Classical baseline | nan               |  0.114792  |   0.148246  |  0.296835   |       0.564677 |        0.517718 |
| Classical baseline | nan               |  0.116701  |   0.149767  |  0.282332   |       0.554375 |        0.506094 |
| Classical baseline | nan               |  0.110241  |   0.143514  |  0.341005   |       0.589939 |        0.535886 |
| Classical baseline | nan               |  0.0983714 |   0.146923  | -0.034438   |       0.33381  |        0.307632 |
| Classical baseline | nan               |  0.0986879 |   0.147774  | -0.0464499  |       0.339153 |        0.310046 |
| Classical baseline | nan               |  0.0984117 |   0.14865   | -0.0588913  |       0.342703 |        0.36428  |

## 5. Best model per target

| target            | best_by_r2         |   best_r2 | best_by_spearman   |   best_spearman |
|:------------------|:-------------------|----------:|:-------------------|----------------:|
| target_excitement | TransformerEncoder | 0.435636  | TransformerEncoder |        0.62434  |
| target_relaxation | TransformerEncoder | 0.329365  | TransformerEncoder |        0.580842 |
| target_engagement | LSTM               | 0.128741  | TransformerEncoder |        0.448832 |
| target_interest   | TransformerEncoder | 0.214179  | TransformerEncoder |        0.429093 |
| target_focus      | TransformerEncoder | 0.170508  | TransformerEncoder |        0.472063 |
| target_stress     | TransformerEncoder | 0.148657  | TransformerEncoder |        0.450752 |
| target_attention  | TransformerEncoder | 0.0532111 | TransformerEncoder |        0.435877 |

## 6. Main findings

The MLP model serves as a neural tabular baseline. It confirms that simply replacing classical models with a fully connected network is not sufficient for this task.

The LSTM model improves the results for several targets by using neighboring EEG/POW windows. This supports the hypothesis that local temporal dynamics are informative for PM prediction.

The TransformerEncoder provides the strongest DL result in the current project. It improves several targets compared with the single-window classical baseline, especially `target_excitement`, `target_relaxation`, `target_interest`, and `target_focus`.

Some targets, such as `target_engagement` and `target_attention`, are still better predicted by classical models. This suggests that the benefit of temporal attention is target-dependent.

## 7. Autoencoder generation block

The Autoencoder was trained on standardized EEG/POW feature vectors.

| Metric | Value |
|---|---:|
| `scaled_reconstruction_mse` | 0.145195 |
| `scaled_reconstruction_rmse` | 0.381044 |
| `scaled_reconstruction_mae` | 0.103780 |
| `scaled_reconstruction_cosine_mean` | 0.778328 |
| `scaled_feature_mean_abs_diff_mean` | 0.031088 |
| `scaled_feature_std_abs_diff_mean` | 0.148859 |
| `scaled_real_synthetic_pairwise_cosine_mean` | 0.105439 |

The generation is performed in EEG/POW feature space rather than raw EEG waveform space. Synthetic vectors are obtained by perturbing latent representations and decoding them. This is a lightweight approximation suitable for the current exam project.

## 8. Limitations

- The project uses device-derived PM targets rather than independent expert labels.
- Synthetic generation is performed in feature space, not directly in raw EEG signal space.
- GroupKFold by subject is much stricter than random split and exposes inter-subject variability.
- The current models do not explicitly model electrode topology.

## 9. Next steps

- Add a cleaner final notebook for demonstration.
- Add a short defense text.
- Optionally add error-based anomaly detection from model residuals.
- Optionally test synthetic augmentation in a downstream predictor.