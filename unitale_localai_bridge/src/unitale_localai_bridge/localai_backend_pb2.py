"""由本仓库精简 LocalAI wire schema 创建的 protobuf message 类。"""

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def _build_descriptor():
    """构造与 LocalAI v4.11.0 固定 backend.proto 字段号兼容的描述符。"""
    file_descriptor = descriptor_pb2.FileDescriptorProto(
        name="unitale_localai_backend.proto",
        package="backend",
        syntax="proto3",
    )

    def add_message(name, fields):
        message = file_descriptor.message_type.add(name=name)
        for field_name, number, field_type in fields:
            message.field.add(
                name=field_name,
                number=number,
                label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
                type=field_type,
            )
        return message

    string_type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    bytes_type = descriptor_pb2.FieldDescriptorProto.TYPE_BYTES
    bool_type = descriptor_pb2.FieldDescriptorProto.TYPE_BOOL
    enum_type = descriptor_pb2.FieldDescriptorProto.TYPE_ENUM

    add_message("HealthMessage", [])
    add_message("PredictOptions", [("Prompt", 1, string_type), ("ModelIdentity", 54, string_type)])
    add_message("Reply", [("message", 1, bytes_type)])
    add_message("ModelOptions", [("Model", 1, string_type)])
    add_message("Result", [("message", 1, string_type), ("success", 2, bool_type)])

    status = add_message("StatusResponse", [])
    state = status.enum_type.add(name="State")
    for name, number in (("UNINITIALIZED", 0), ("BUSY", 1), ("READY", 2), ("ERROR", -1)):
        state.value.add(name=name, number=number)
    status.field.add(
        name="state",
        number=1,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
        type=enum_type,
        type_name=".backend.StatusResponse.State",
    )

    service = file_descriptor.service.add(name="Backend")
    methods = (
        ("Health", "HealthMessage", "Reply"),
        ("Free", "HealthMessage", "Result"),
        ("Predict", "PredictOptions", "Reply"),
        ("LoadModel", "ModelOptions", "Result"),
        ("Status", "HealthMessage", "StatusResponse"),
    )
    for method_name, request_name, response_name in methods:
        service.method.add(
            name=method_name,
            input_type=f".backend.{request_name}",
            output_type=f".backend.{response_name}",
        )
    return descriptor_pool.Default().Add(file_descriptor)


DESCRIPTOR = _build_descriptor()
HealthMessage = message_factory.GetMessageClass(DESCRIPTOR.message_types_by_name["HealthMessage"])
PredictOptions = message_factory.GetMessageClass(DESCRIPTOR.message_types_by_name["PredictOptions"])
Reply = message_factory.GetMessageClass(DESCRIPTOR.message_types_by_name["Reply"])
ModelOptions = message_factory.GetMessageClass(DESCRIPTOR.message_types_by_name["ModelOptions"])
Result = message_factory.GetMessageClass(DESCRIPTOR.message_types_by_name["Result"])
StatusResponse = message_factory.GetMessageClass(DESCRIPTOR.message_types_by_name["StatusResponse"])

__all__ = [
    "DESCRIPTOR",
    "HealthMessage",
    "ModelOptions",
    "PredictOptions",
    "Reply",
    "Result",
    "StatusResponse",
]
