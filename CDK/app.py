#!/usr/bin/env python3
import aws_cdk as cdk
from infra_stack import ECGStack

app = cdk.App()
ECGStack(app, "ECGStack")
app.synth()
