import os, json, boto3

ssm = boto3.client("ssm")

REGION_PARAM = os.environ["BEDROCK_REGION_PARAM"]    # p.ej. "/ecg-app/BEDROCK_REGION"
MODEL_PARAM  = os.environ["BEDROCK_MODEL_ID_PARAM"]  # p.ej. "/ecg-app/BEDROCK_MODEL_ID"

def _param(name): 
    return ssm.get_parameter(Name=name)["Parameter"]["Value"]

def handler(event, context):
    try:
        body = event.get("body") or "{}"
        if isinstance(body, str): body = json.loads(body)

        prediction = body.get("prediction", {})
        region = _param(REGION_PARAM)
        model_id = _param(MODEL_PARAM)

        bedrock = boto3.client("bedrock-runtime", region_name=region)

        prompt = (
            "Eres cardiólogo. Devuelve un JSON con campos 'summary','risk','recommendations' "
            f"en español para la predicción: {json.dumps(prediction, ensure_ascii=False)}"
        )

        req = {
            "messages":[{"role":"user","content":[{"type":"text","text": prompt}]}],
            "max_tokens":512
        }

        resp = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(req)
        )
        out = json.loads(resp["body"].read())

        return {"statusCode":200, "headers":{"Content-Type":"application/json"},
                "body": json.dumps({"ok": True, "explanation": out})}
    except Exception as e:
        return {"statusCode":500, "body": json.dumps({"ok": False, "error": str(e)})}
