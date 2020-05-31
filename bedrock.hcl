version = "1.0"

train {
    step generate_features {
        image = "basisai/workload-standard:v0.1.2"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements-train.txt",
        ]
        script = [{sh = ["python3 generate_features.py"]}]
        resources {
            cpu = "4"
            memory = "30G"
        }
    }

    step train {
        image = "basisai/workload-standard:v0.1.2"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements-train.txt",
        ]
        script = [{sh = ["python3 train.py"]}]
        resources {
            cpu = "4"
            memory = "32G"
        }
        depends_on = ["generate_features"]
    }

    parameters {
        MODEL_VER = "lightgbm"
        LR = "0.05"
    }
}

serve {
    image = "python:3.7"
    install = [
        "pip3 install --upgrade pip",
        "pip3 install -r requirements-serve.txt",
    ]
    script = [
        {sh = [
            "gunicorn --bind=:${BEDROCK_SERVER_PORT:-8080} --worker-class=gthread --workers=${WORKERS} --timeout=300 --preload serve_http:app"
        ]}
    ]

    parameters {
        WORKERS = "1"
    }
}
