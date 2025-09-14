import os, json, boto3

ssm = boto3.client("ssm")
smr = boto3.client("sagemaker-runtime")

SM_PARAM = os.environ["SM_ENDPOINT_PARAM"]  # p.ej. "/ecg-app/SM_ENDPOINT_NAME"

def _endpoint():
    return ssm.get_parameter(Name=SM_PARAM)["Parameter"]["Value"]

def handler(event, context):
    try:
        body = event.get("body") or "{}"
        if isinstance(body, str): body = json.loads(body)

        # Espera: { "image_base64": "..." } u opciones de inference.py
        payload = json.dumps(body).encode("utf-8")
        resp = smr.invoke_endpoint(
            EndpointName=_endpoint(),
            ContentType="application/json",
            Body=payload
        )
        result = json.loads(resp["Body"].read().decode("utf-8"))
        return {"statusCode":200, "headers":{"Content-Type":"application/json"},
                "body": json.dumps({"ok": True, "prediction": result})}
    except Exception as e:
        return {"statusCode":500, "body": json.dumps({"ok": False, "error": str(e)})}
