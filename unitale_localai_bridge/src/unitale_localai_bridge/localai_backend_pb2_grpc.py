"""LocalAI v4.11.0 Backend gRPC service path bindings for the wire subset."""

import grpc

from . import localai_backend_pb2 as localai__backend__pb2


class BackendStub:
    """客户端绑定，供本机协议测试使用。"""

    def __init__(self, channel):
        self.Health = channel.unary_unary(
            "/backend.Backend/Health",
            request_serializer=localai__backend__pb2.HealthMessage.SerializeToString,
            response_deserializer=localai__backend__pb2.Reply.FromString,
        )
        self.Free = channel.unary_unary(
            "/backend.Backend/Free",
            request_serializer=localai__backend__pb2.HealthMessage.SerializeToString,
            response_deserializer=localai__backend__pb2.Result.FromString,
        )
        self.Predict = channel.unary_unary(
            "/backend.Backend/Predict",
            request_serializer=localai__backend__pb2.PredictOptions.SerializeToString,
            response_deserializer=localai__backend__pb2.Reply.FromString,
        )
        self.LoadModel = channel.unary_unary(
            "/backend.Backend/LoadModel",
            request_serializer=localai__backend__pb2.ModelOptions.SerializeToString,
            response_deserializer=localai__backend__pb2.Result.FromString,
        )
        self.Status = channel.unary_unary(
            "/backend.Backend/Status",
            request_serializer=localai__backend__pb2.HealthMessage.SerializeToString,
            response_deserializer=localai__backend__pb2.StatusResponse.FromString,
        )


class BackendServicer:
    """servicer 基类占位，具体 handler 位于 server.py。"""


def add_BackendServicer_to_server(servicer, server):
    """将 fixed Backend RPC 子集注册到 grpc.Server。"""
    handlers = {
        "Health": grpc.unary_unary_rpc_method_handler(
            servicer.Health,
            request_deserializer=localai__backend__pb2.HealthMessage.FromString,
            response_serializer=localai__backend__pb2.Reply.SerializeToString,
        ),
        "Free": grpc.unary_unary_rpc_method_handler(
            servicer.Free,
            request_deserializer=localai__backend__pb2.HealthMessage.FromString,
            response_serializer=localai__backend__pb2.Result.SerializeToString,
        ),
        "Predict": grpc.unary_unary_rpc_method_handler(
            servicer.Predict,
            request_deserializer=localai__backend__pb2.PredictOptions.FromString,
            response_serializer=localai__backend__pb2.Reply.SerializeToString,
        ),
        "LoadModel": grpc.unary_unary_rpc_method_handler(
            servicer.LoadModel,
            request_deserializer=localai__backend__pb2.ModelOptions.FromString,
            response_serializer=localai__backend__pb2.Result.SerializeToString,
        ),
        "Status": grpc.unary_unary_rpc_method_handler(
            servicer.Status,
            request_deserializer=localai__backend__pb2.HealthMessage.FromString,
            response_serializer=localai__backend__pb2.StatusResponse.SerializeToString,
        ),
    }
    server.add_generic_rpc_handlers(
        (grpc.method_handlers_generic_handler("backend.Backend", handlers),)
    )
