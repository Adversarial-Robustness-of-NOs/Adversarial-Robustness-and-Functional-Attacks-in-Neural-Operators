set -x

python -m temporal.train --problem temporal_ns_2d --model_config=temporal/ns_2d/fno_model.toml --data_config=temporal/ns_2d/data.toml --output_model=temporal/ns_2d/trained_models/best_fno_model.pth
python -m temporal.train --problem temporal_ns_2d --model_config=temporal/ns_2d/ffno_model.toml --data_config=temporal/ns_2d/data.toml --output_model=temporal/ns_2d/trained_models/best_ffno_model.pth
python -m temporal.train --problem temporal_ns_2d --model_config=temporal/ns_2d/cno_model.toml --data_config=temporal/ns_2d/data.toml --output_model=temporal/ns_2d/trained_models/best_cno_model.pth

set +x