# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: submission.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import geom_pb2 as geom__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='submission.proto',
  package='neurips_dataset',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x10submission.proto\x12\x0fneurips_dataset\x1a\ngeom.proto\":\n\nTrajectory\x12,\n\ntrajectory\x18\x01 \x03(\x0b\x32\x18.neurips_dataset.Vector3\"U\n\x12WeightedTrajectory\x12/\n\ntrajectory\x18\x01 \x01(\x0b\x32\x1b.neurips_dataset.Trajectory\x12\x0e\n\x06weight\x18\x02 \x01(\x02\"\x8e\x01\n\x10ObjectPrediction\x12\x10\n\x08track_id\x18\x01 \x01(\x04\x12\x10\n\x08scene_id\x18\x02 \x01(\t\x12\x39\n\x0ctrajectories\x18\x03 \x03(\x0b\x32#.neurips_dataset.WeightedTrajectory\x12\x1b\n\x13uncertainty_measure\x18\x04 \x01(\x02\"D\n\nSubmission\x12\x36\n\x0bpredictions\x18\x01 \x03(\x0b\x32!.neurips_dataset.ObjectPredictionb\x06proto3'
  ,
  dependencies=[geom__pb2.DESCRIPTOR,])




_TRAJECTORY = _descriptor.Descriptor(
  name='Trajectory',
  full_name='neurips_dataset.Trajectory',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='trajectory', full_name='neurips_dataset.Trajectory.trajectory', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=49,
  serialized_end=107,
)


_WEIGHTEDTRAJECTORY = _descriptor.Descriptor(
  name='WeightedTrajectory',
  full_name='neurips_dataset.WeightedTrajectory',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='trajectory', full_name='neurips_dataset.WeightedTrajectory.trajectory', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='weight', full_name='neurips_dataset.WeightedTrajectory.weight', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=109,
  serialized_end=194,
)


_OBJECTPREDICTION = _descriptor.Descriptor(
  name='ObjectPrediction',
  full_name='neurips_dataset.ObjectPrediction',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='track_id', full_name='neurips_dataset.ObjectPrediction.track_id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='scene_id', full_name='neurips_dataset.ObjectPrediction.scene_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='trajectories', full_name='neurips_dataset.ObjectPrediction.trajectories', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='uncertainty_measure', full_name='neurips_dataset.ObjectPrediction.uncertainty_measure', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=197,
  serialized_end=339,
)


_SUBMISSION = _descriptor.Descriptor(
  name='Submission',
  full_name='neurips_dataset.Submission',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='predictions', full_name='neurips_dataset.Submission.predictions', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=341,
  serialized_end=409,
)

_TRAJECTORY.fields_by_name['trajectory'].message_type = geom__pb2._VECTOR3
_WEIGHTEDTRAJECTORY.fields_by_name['trajectory'].message_type = _TRAJECTORY
_OBJECTPREDICTION.fields_by_name['trajectories'].message_type = _WEIGHTEDTRAJECTORY
_SUBMISSION.fields_by_name['predictions'].message_type = _OBJECTPREDICTION
DESCRIPTOR.message_types_by_name['Trajectory'] = _TRAJECTORY
DESCRIPTOR.message_types_by_name['WeightedTrajectory'] = _WEIGHTEDTRAJECTORY
DESCRIPTOR.message_types_by_name['ObjectPrediction'] = _OBJECTPREDICTION
DESCRIPTOR.message_types_by_name['Submission'] = _SUBMISSION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Trajectory = _reflection.GeneratedProtocolMessageType('Trajectory', (_message.Message,), {
  'DESCRIPTOR' : _TRAJECTORY,
  '__module__' : 'submission_pb2'
  # @@protoc_insertion_point(class_scope:neurips_dataset.Trajectory)
  })
_sym_db.RegisterMessage(Trajectory)

WeightedTrajectory = _reflection.GeneratedProtocolMessageType('WeightedTrajectory', (_message.Message,), {
  'DESCRIPTOR' : _WEIGHTEDTRAJECTORY,
  '__module__' : 'submission_pb2'
  # @@protoc_insertion_point(class_scope:neurips_dataset.WeightedTrajectory)
  })
_sym_db.RegisterMessage(WeightedTrajectory)

ObjectPrediction = _reflection.GeneratedProtocolMessageType('ObjectPrediction', (_message.Message,), {
  'DESCRIPTOR' : _OBJECTPREDICTION,
  '__module__' : 'submission_pb2'
  # @@protoc_insertion_point(class_scope:neurips_dataset.ObjectPrediction)
  })
_sym_db.RegisterMessage(ObjectPrediction)

Submission = _reflection.GeneratedProtocolMessageType('Submission', (_message.Message,), {
  'DESCRIPTOR' : _SUBMISSION,
  '__module__' : 'submission_pb2'
  # @@protoc_insertion_point(class_scope:neurips_dataset.Submission)
  })
_sym_db.RegisterMessage(Submission)


# @@protoc_insertion_point(module_scope)
