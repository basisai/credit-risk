version = "1.0"

train {
    step preproc_agg {
        image = "basisai/workload-standard:v0.1.2"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements-train.txt",
        ]
        script = [{sh = ["python3 tasks/preproc_agg.py"]}]
        resources {
            cpu = "0.5"
            memory = "1G"
        }
    }

    step features_fs {
        image = "basisai/workload-standard:v0.1.2"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements-train.txt",
        ]
        script = [{sh = ["python3 tasks/features_fs.py"]}]
        resources {
            cpu = "0.5"
            memory = "1G"
        }
        depends_on = ["preproc_agg"]
    }

    parameters {
        EXECUTION_DATE = ""
    }
}
