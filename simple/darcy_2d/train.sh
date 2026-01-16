set -x
python -m simple.train --problem="simple_darcy_2d" --model_config=simple/darcy_2d/fno_model.toml --data_config=simple/darcy_2d/train_data.toml --output_model=simple/darcy_2d/trained_models/best_fno_model.pth
python -m simple.train --problem="simple_darcy_2d" --model_config=simple/darcy_2d/ffno_model.toml --data_config=simple/darcy_2d/train_data.toml --output_model=simple/darcy_2d/trained_models/best_ffno_model.pth
python -m simple.train --problem="simple_darcy_2d" --model_config=simple/darcy_2d/cno_model.toml --data_config=simple/darcy_2d/train_data.toml --output_model=simple/darcy_2d/trained_models/best_cno_model.pth
set +x