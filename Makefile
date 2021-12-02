# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* Twitter_bot_detection_713/*.py

black:
	@black scripts/* Twitter_bot_detection_713/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr Twitter_bot_detection_713-*.dist-info
	@rm -fr Twitter_bot_detection_713.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

###Streamlit command

run_streamlit:
	streamlit run Twitter_bot_detection_713/app.py


####GCP make commands

LOCAL_PATH='Twitter_bot_detection_713/data/pickled_data/y_test_25.pickle'

PROJECT_ID=astute-arcanum-332414

BUCKET_NAME=tweet-project-713

BUCKET_FOLDER=data

BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

BUCKET_TRAINING_FOLDER = 'trainings'

PACKAGE_NAME=Twitter_bot_detection_713

FILENAME=trainer_text

JOB_NAME=Twitter_bot_detection_models_$(shell date +'%Y%m%d_%H%M%S')

REGION=us-central1

PYTHON_VERSION=3.7

FRAMEWORK=scikit-learn

RUNTIME_VERSION=1.15

upload_data:
	-@gsutil cp ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}



run_streamlit:
	streamlit run app.py

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs


##--scale-tier custom \
		--master-machine-type n1-highmem-32 \
