## Running Temporal Attacks

To run the autoregressive attacks, you first need to generate samples on which the attack is performed. Before you generate the data, go to `temporal/ns_2d/data/generate_ns_data_legacy.py` and change the code there such that its output is saved to `temporal/ns_2d/data/attack_data.h5`. Optionally, you can reduce the number of generated samples to 100 or 1000. Then run
    
    python -m temporal.ns_2d.data.generate_ns_data_legacy

To perform an attack maximizing the error of the entier time series, run

    python -m temporal.ns_2d.attacks --attack_batch_size 10 --no_smoothing --n_samples 100 --pgd_steps 100 --model_path temporal/ns_2d/trained_models/best_fno_model.pth --config temporal/ns_2d/fno_model.toml --output_dir temporal/ns_2d/data/fno_attack_all
    python -m temporal.ns_2d.attacks --attack_batch_size 10 --no_smoothing --n_samples 100 --pgd_steps 100 --model_path temporal/ns_2d/trained_models/best_fno_model.pth --config temporal/ns_2d/fno_model.toml --output_dir temporal/ns_2d/data/fno_attack_all

If you only want to maximize the error of the last time step, run

    python -m temporal.ns_2d.attacks --attack_batch_size 1 --no_smoothing --n_samples 10 --pgd_steps 100 --model_path temporal/ns_2d/trained_models/best_fno_model.pth --config temporal/ns_2d/fno_model.toml --output_dir temporal/ns_2d/data/fno_attack_last --loss last

To evaluate the samples and create the plots, use

    python -m temporal.ns_2d.evaluate_temporal --config temporal/ns_2d/fno_model.toml --model_path temporal/ns_2d/trained_models/best_fno_model.pth --data_path temporal/ns_2d/data/fno_attack_all/eval_pgd_spatial.h5 --output_dir temporal/ns_2d/data/fno_attack_all/spatial/ --batch_size 1
    python -m temporal.ns_2d.evaluate_temporal --config temporal/ns_2d/fno_model.toml --model_path temporal/ns_2d/trained_models/best_fno_model.pth --data_path temporal/ns_2d/data/fno_attack_all/eval_pgd_spectral.h5 --output_dir temporal/ns_2d/data/fno_attack_all/spectral/ --batch_size 1
    python -m temporal.ns_2d.evaluate_temporal --config temporal/ns_2d/fno_model.toml --model_path temporal/ns_2d/trained_models/best_fno_model.pth --data_path temporal/ns_2d/data/fno_attack_all/eval_pgd_spatial_pi.h5 --output_dir temporal/ns_2d/data/fno_attack_all/spatial_pi/ --batch_size 1
    python -m temporal.ns_2d.evaluate_temporal --config temporal/ns_2d/fno_model.toml --model_path temporal/ns_2d/trained_models/best_fno_model.pth --data_path temporal/ns_2d/data/fno_attack_all/eval_pgd_spectral_pi.h5 --output_dir temporal/ns_2d/data/fno_attack_all/spectral_pi/ --batch_size 1