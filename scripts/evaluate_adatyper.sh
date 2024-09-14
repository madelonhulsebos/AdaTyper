# To lookup from data used for training model
data_identifier=19
# To lookup from the trained model
pipeline_identifier=19
# These evaluation functions depend on the preset thresholds
function_1='configured'
function_2='ctu'

python evaluation/predictors_evaluation.py \
    --function ${function_1} \
    --data_identifier ${data_identifier} \
    --pipeline_identifier ${data_identifier} \

python evaluation/predictors_evaluation.py \
    --function ${function_2} \
    --data_identifier ${data_identifier} \
    --pipeline_identifier ${data_identifier} \