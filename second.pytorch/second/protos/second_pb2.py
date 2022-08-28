# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: second/protos/second.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from second.protos import losses_pb2 as second_dot_protos_dot_losses__pb2
from second.protos import box_coder_pb2 as second_dot_protos_dot_box__coder__pb2
from second.protos import target_pb2 as second_dot_protos_dot_target__pb2
from second.protos import voxel_generator_pb2 as second_dot_protos_dot_voxel__generator__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='second/protos/second.proto',
  package='second.protos',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x1asecond/protos/second.proto\x12\rsecond.protos\x1a\x1asecond/protos/losses.proto\x1a\x1dsecond/protos/box_coder.proto\x1a\x1asecond/protos/target.proto\x1a#second/protos/voxel_generator.proto\"\x85\x0c\n\x08VoxelNet\x12\x1a\n\x12network_class_name\x18\x01 \x01(\t\x12\x36\n\x0fvoxel_generator\x18\x02 \x01(\x0b\x32\x1d.second.protos.VoxelGenerator\x12N\n\x17voxel_feature_extractor\x18\x03 \x01(\x0b\x32-.second.protos.VoxelNet.VoxelFeatureExtractor\x12P\n\x18middle_feature_extractor\x18\x04 \x01(\x0b\x32..second.protos.VoxelNet.MiddleFeatureExtractor\x12(\n\x03rpn\x18\x05 \x01(\x0b\x32\x1b.second.protos.VoxelNet.RPN\x12\x1a\n\x12num_point_features\x18\x06 \x01(\r\x12\x19\n\x11use_sigmoid_score\x18\x07 \x01(\x08\x12!\n\x04loss\x18\x08 \x01(\x0b\x32\x13.second.protos.Loss\x12\x1f\n\x17\x65ncode_rad_error_by_sin\x18\t \x01(\x08\x12\"\n\x1a\x65ncode_background_as_zeros\x18\n \x01(\x08\x12 \n\x18use_direction_classifier\x18\x0b \x01(\x08\x12\x1d\n\x15\x64irection_loss_weight\x18\x0c \x01(\x02\x12\x18\n\x10pos_class_weight\x18\r \x01(\x02\x12\x18\n\x10neg_class_weight\x18\x0e \x01(\x02\x12<\n\x0eloss_norm_type\x18\x0f \x01(\x0e\x32$.second.protos.VoxelNet.LossNormType\x12*\n\tbox_coder\x18\x10 \x01(\x0b\x32\x17.second.protos.BoxCoder\x12\x36\n\x0ftarget_assigner\x18\x11 \x01(\x0b\x32\x1d.second.protos.TargetAssigner\x12\x1f\n\x17post_center_limit_range\x18\x12 \x03(\x02\x12\x18\n\x10\x64irection_offset\x18\x13 \x01(\x02\x12\x18\n\x10sin_error_factor\x18\x14 \x01(\x02\x12\x1a\n\x12nms_class_agnostic\x18\x15 \x01(\x08\x12\x1a\n\x12num_direction_bins\x18\x16 \x01(\x03\x12\x1e\n\x16\x64irection_limit_offset\x18\x17 \x01(\x02\x12\x13\n\x0blidar_input\x18\x18 \x01(\x08\x1az\n\x15VoxelFeatureExtractor\x12\x19\n\x11module_class_name\x18\x01 \x01(\t\x12\x13\n\x0bnum_filters\x18\x02 \x03(\x05\x12\x15\n\rwith_distance\x18\x03 \x01(\x08\x12\x1a\n\x12num_input_features\x18\x04 \x01(\x05\x1a\xa0\x01\n\x16MiddleFeatureExtractor\x12\x19\n\x11module_class_name\x18\x01 \x01(\t\x12\x19\n\x11num_filters_down1\x18\x02 \x03(\x05\x12\x19\n\x11num_filters_down2\x18\x03 \x03(\x05\x12\x1a\n\x12num_input_features\x18\x04 \x01(\x05\x12\x19\n\x11\x64ownsample_factor\x18\x05 \x01(\x05\x1a\xdf\x01\n\x03RPN\x12\x19\n\x11module_class_name\x18\x01 \x01(\t\x12\x12\n\nlayer_nums\x18\x02 \x03(\x05\x12\x15\n\rlayer_strides\x18\x03 \x03(\x05\x12\x13\n\x0bnum_filters\x18\x04 \x03(\x05\x12\x18\n\x10upsample_strides\x18\x05 \x03(\x01\x12\x1c\n\x14num_upsample_filters\x18\x06 \x03(\x05\x12\x15\n\ruse_groupnorm\x18\x07 \x01(\x08\x12\x12\n\nnum_groups\x18\x08 \x01(\x05\x12\x1a\n\x12num_input_features\x18\t \x01(\x05\"`\n\x0cLossNormType\x12\x15\n\x11NormByNumExamples\x10\x00\x12\x16\n\x12NormByNumPositives\x10\x01\x12\x13\n\x0fNormByNumPosNeg\x10\x02\x12\x0c\n\x08\x44ontNorm\x10\x03\x62\x06proto3')
  ,
  dependencies=[second_dot_protos_dot_losses__pb2.DESCRIPTOR,second_dot_protos_dot_box__coder__pb2.DESCRIPTOR,second_dot_protos_dot_target__pb2.DESCRIPTOR,second_dot_protos_dot_voxel__generator__pb2.DESCRIPTOR,])



_VOXELNET_LOSSNORMTYPE = _descriptor.EnumDescriptor(
  name='LossNormType',
  full_name='second.protos.VoxelNet.LossNormType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NormByNumExamples', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NormByNumPositives', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NormByNumPosNeg', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DontNorm', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1615,
  serialized_end=1711,
)
_sym_db.RegisterEnumDescriptor(_VOXELNET_LOSSNORMTYPE)


_VOXELNET_VOXELFEATUREEXTRACTOR = _descriptor.Descriptor(
  name='VoxelFeatureExtractor',
  full_name='second.protos.VoxelNet.VoxelFeatureExtractor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='module_class_name', full_name='second.protos.VoxelNet.VoxelFeatureExtractor.module_class_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_filters', full_name='second.protos.VoxelNet.VoxelFeatureExtractor.num_filters', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='with_distance', full_name='second.protos.VoxelNet.VoxelFeatureExtractor.with_distance', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_input_features', full_name='second.protos.VoxelNet.VoxelFeatureExtractor.num_input_features', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
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
  serialized_start=1102,
  serialized_end=1224,
)

_VOXELNET_MIDDLEFEATUREEXTRACTOR = _descriptor.Descriptor(
  name='MiddleFeatureExtractor',
  full_name='second.protos.VoxelNet.MiddleFeatureExtractor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='module_class_name', full_name='second.protos.VoxelNet.MiddleFeatureExtractor.module_class_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_filters_down1', full_name='second.protos.VoxelNet.MiddleFeatureExtractor.num_filters_down1', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_filters_down2', full_name='second.protos.VoxelNet.MiddleFeatureExtractor.num_filters_down2', index=2,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_input_features', full_name='second.protos.VoxelNet.MiddleFeatureExtractor.num_input_features', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='downsample_factor', full_name='second.protos.VoxelNet.MiddleFeatureExtractor.downsample_factor', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
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
  serialized_start=1227,
  serialized_end=1387,
)

_VOXELNET_RPN = _descriptor.Descriptor(
  name='RPN',
  full_name='second.protos.VoxelNet.RPN',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='module_class_name', full_name='second.protos.VoxelNet.RPN.module_class_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='layer_nums', full_name='second.protos.VoxelNet.RPN.layer_nums', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='layer_strides', full_name='second.protos.VoxelNet.RPN.layer_strides', index=2,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_filters', full_name='second.protos.VoxelNet.RPN.num_filters', index=3,
      number=4, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='upsample_strides', full_name='second.protos.VoxelNet.RPN.upsample_strides', index=4,
      number=5, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_upsample_filters', full_name='second.protos.VoxelNet.RPN.num_upsample_filters', index=5,
      number=6, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_groupnorm', full_name='second.protos.VoxelNet.RPN.use_groupnorm', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_groups', full_name='second.protos.VoxelNet.RPN.num_groups', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_input_features', full_name='second.protos.VoxelNet.RPN.num_input_features', index=8,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
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
  serialized_start=1390,
  serialized_end=1613,
)

_VOXELNET = _descriptor.Descriptor(
  name='VoxelNet',
  full_name='second.protos.VoxelNet',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='network_class_name', full_name='second.protos.VoxelNet.network_class_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='voxel_generator', full_name='second.protos.VoxelNet.voxel_generator', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='voxel_feature_extractor', full_name='second.protos.VoxelNet.voxel_feature_extractor', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='middle_feature_extractor', full_name='second.protos.VoxelNet.middle_feature_extractor', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rpn', full_name='second.protos.VoxelNet.rpn', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_point_features', full_name='second.protos.VoxelNet.num_point_features', index=5,
      number=6, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_sigmoid_score', full_name='second.protos.VoxelNet.use_sigmoid_score', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='loss', full_name='second.protos.VoxelNet.loss', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='encode_rad_error_by_sin', full_name='second.protos.VoxelNet.encode_rad_error_by_sin', index=8,
      number=9, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='encode_background_as_zeros', full_name='second.protos.VoxelNet.encode_background_as_zeros', index=9,
      number=10, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_direction_classifier', full_name='second.protos.VoxelNet.use_direction_classifier', index=10,
      number=11, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='direction_loss_weight', full_name='second.protos.VoxelNet.direction_loss_weight', index=11,
      number=12, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pos_class_weight', full_name='second.protos.VoxelNet.pos_class_weight', index=12,
      number=13, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='neg_class_weight', full_name='second.protos.VoxelNet.neg_class_weight', index=13,
      number=14, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='loss_norm_type', full_name='second.protos.VoxelNet.loss_norm_type', index=14,
      number=15, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='box_coder', full_name='second.protos.VoxelNet.box_coder', index=15,
      number=16, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='target_assigner', full_name='second.protos.VoxelNet.target_assigner', index=16,
      number=17, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='post_center_limit_range', full_name='second.protos.VoxelNet.post_center_limit_range', index=17,
      number=18, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='direction_offset', full_name='second.protos.VoxelNet.direction_offset', index=18,
      number=19, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sin_error_factor', full_name='second.protos.VoxelNet.sin_error_factor', index=19,
      number=20, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nms_class_agnostic', full_name='second.protos.VoxelNet.nms_class_agnostic', index=20,
      number=21, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_direction_bins', full_name='second.protos.VoxelNet.num_direction_bins', index=21,
      number=22, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='direction_limit_offset', full_name='second.protos.VoxelNet.direction_limit_offset', index=22,
      number=23, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='lidar_input', full_name='second.protos.VoxelNet.lidar_input', index=23,
      number=24, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_VOXELNET_VOXELFEATUREEXTRACTOR, _VOXELNET_MIDDLEFEATUREEXTRACTOR, _VOXELNET_RPN, ],
  enum_types=[
    _VOXELNET_LOSSNORMTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=170,
  serialized_end=1711,
)

_VOXELNET_VOXELFEATUREEXTRACTOR.containing_type = _VOXELNET
_VOXELNET_MIDDLEFEATUREEXTRACTOR.containing_type = _VOXELNET
_VOXELNET_RPN.containing_type = _VOXELNET
_VOXELNET.fields_by_name['voxel_generator'].message_type = second_dot_protos_dot_voxel__generator__pb2._VOXELGENERATOR
_VOXELNET.fields_by_name['voxel_feature_extractor'].message_type = _VOXELNET_VOXELFEATUREEXTRACTOR
_VOXELNET.fields_by_name['middle_feature_extractor'].message_type = _VOXELNET_MIDDLEFEATUREEXTRACTOR
_VOXELNET.fields_by_name['rpn'].message_type = _VOXELNET_RPN
_VOXELNET.fields_by_name['loss'].message_type = second_dot_protos_dot_losses__pb2._LOSS
_VOXELNET.fields_by_name['loss_norm_type'].enum_type = _VOXELNET_LOSSNORMTYPE
_VOXELNET.fields_by_name['box_coder'].message_type = second_dot_protos_dot_box__coder__pb2._BOXCODER
_VOXELNET.fields_by_name['target_assigner'].message_type = second_dot_protos_dot_target__pb2._TARGETASSIGNER
_VOXELNET_LOSSNORMTYPE.containing_type = _VOXELNET
DESCRIPTOR.message_types_by_name['VoxelNet'] = _VOXELNET
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

VoxelNet = _reflection.GeneratedProtocolMessageType('VoxelNet', (_message.Message,), dict(

  VoxelFeatureExtractor = _reflection.GeneratedProtocolMessageType('VoxelFeatureExtractor', (_message.Message,), dict(
    DESCRIPTOR = _VOXELNET_VOXELFEATUREEXTRACTOR,
    __module__ = 'second.protos.second_pb2'
    # @@protoc_insertion_point(class_scope:second.protos.VoxelNet.VoxelFeatureExtractor)
    ))
  ,

  MiddleFeatureExtractor = _reflection.GeneratedProtocolMessageType('MiddleFeatureExtractor', (_message.Message,), dict(
    DESCRIPTOR = _VOXELNET_MIDDLEFEATUREEXTRACTOR,
    __module__ = 'second.protos.second_pb2'
    # @@protoc_insertion_point(class_scope:second.protos.VoxelNet.MiddleFeatureExtractor)
    ))
  ,

  RPN = _reflection.GeneratedProtocolMessageType('RPN', (_message.Message,), dict(
    DESCRIPTOR = _VOXELNET_RPN,
    __module__ = 'second.protos.second_pb2'
    # @@protoc_insertion_point(class_scope:second.protos.VoxelNet.RPN)
    ))
  ,
  DESCRIPTOR = _VOXELNET,
  __module__ = 'second.protos.second_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.VoxelNet)
  ))
_sym_db.RegisterMessage(VoxelNet)
_sym_db.RegisterMessage(VoxelNet.VoxelFeatureExtractor)
_sym_db.RegisterMessage(VoxelNet.MiddleFeatureExtractor)
_sym_db.RegisterMessage(VoxelNet.RPN)


# @@protoc_insertion_point(module_scope)
