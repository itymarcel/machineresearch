       ЃK"	  РЅкћжAbrain.Event:2<хЎ6Нћ     s@v[	IэЅкћжA"Аї
{
conv1d_1_inputPlaceholder*
dtype0*,
_output_shapes
:џџџџџџџџџЏ*!
shape:џџџџџџџџџЏ
r
conv1d_1/random_uniform/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
`
conv1d_1/random_uniform/minConst*
valueB
 *ьбО*
dtype0*
_output_shapes
: 
`
conv1d_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ьб>
Ў
%conv1d_1/random_uniform/RandomUniformRandomUniformconv1d_1/random_uniform/shape*
T0*
dtype0*"
_output_shapes
:*
seed2ггУ*
seedБџх)
}
conv1d_1/random_uniform/subSubconv1d_1/random_uniform/maxconv1d_1/random_uniform/min*
T0*
_output_shapes
: 

conv1d_1/random_uniform/mulMul%conv1d_1/random_uniform/RandomUniformconv1d_1/random_uniform/sub*
T0*"
_output_shapes
:

conv1d_1/random_uniformAddconv1d_1/random_uniform/mulconv1d_1/random_uniform/min*
T0*"
_output_shapes
:

conv1d_1/kernel
VariableV2*
shared_name *
dtype0*"
_output_shapes
:*
	container *
shape:
Ф
conv1d_1/kernel/AssignAssignconv1d_1/kernelconv1d_1/random_uniform*
use_locking(*
T0*"
_class
loc:@conv1d_1/kernel*
validate_shape(*"
_output_shapes
:

conv1d_1/kernel/readIdentityconv1d_1/kernel*
T0*"
_class
loc:@conv1d_1/kernel*"
_output_shapes
:
[
conv1d_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
y
conv1d_1/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
­
conv1d_1/bias/AssignAssignconv1d_1/biasconv1d_1/Const*
use_locking(*
T0* 
_class
loc:@conv1d_1/bias*
validate_shape(*
_output_shapes
:
t
conv1d_1/bias/readIdentityconv1d_1/bias*
T0* 
_class
loc:@conv1d_1/bias*
_output_shapes
:
l
"conv1d_1/convolution/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
e
#conv1d_1/convolution/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
Љ
conv1d_1/convolution/ExpandDims
ExpandDimsconv1d_1_input#conv1d_1/convolution/ExpandDims/dim*

Tdim0*
T0*0
_output_shapes
:џџџџџџџџџЏ
g
%conv1d_1/convolution/ExpandDims_1/dimConst*
_output_shapes
: *
value	B : *
dtype0
Љ
!conv1d_1/convolution/ExpandDims_1
ExpandDimsconv1d_1/kernel/read%conv1d_1/convolution/ExpandDims_1/dim*&
_output_shapes
:*

Tdim0*
T0

conv1d_1/convolution/Conv2DConv2Dconv1d_1/convolution/ExpandDims!conv1d_1/convolution/ExpandDims_1*
paddingVALID*0
_output_shapes
:џџџџџџџџџЌ*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

conv1d_1/convolution/SqueezeSqueezeconv1d_1/convolution/Conv2D*
T0*,
_output_shapes
:џџџџџџџџџЌ*
squeeze_dims

k
conv1d_1/Reshape/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:

conv1d_1/ReshapeReshapeconv1d_1/bias/readconv1d_1/Reshape/shape*
T0*
Tshape0*"
_output_shapes
:
z
conv1d_1/addAddconv1d_1/convolution/Squeezeconv1d_1/Reshape*,
_output_shapes
:џџџџџџџџџЌ*
T0
Z
conv1d_1/ReluReluconv1d_1/add*,
_output_shapes
:џџџџџџџџџЌ*
T0
r
conv1d_2/random_uniform/shapeConst*!
valueB"      @   *
dtype0*
_output_shapes
:
`
conv1d_2/random_uniform/minConst*
valueB
 *ьбН*
dtype0*
_output_shapes
: 
`
conv1d_2/random_uniform/maxConst*
valueB
 *ьб=*
dtype0*
_output_shapes
: 
Ў
%conv1d_2/random_uniform/RandomUniformRandomUniformconv1d_2/random_uniform/shape*
seedБџх)*
T0*
dtype0*"
_output_shapes
:@*
seed2пы
}
conv1d_2/random_uniform/subSubconv1d_2/random_uniform/maxconv1d_2/random_uniform/min*
_output_shapes
: *
T0

conv1d_2/random_uniform/mulMul%conv1d_2/random_uniform/RandomUniformconv1d_2/random_uniform/sub*
T0*"
_output_shapes
:@

conv1d_2/random_uniformAddconv1d_2/random_uniform/mulconv1d_2/random_uniform/min*
T0*"
_output_shapes
:@

conv1d_2/kernel
VariableV2*
shape:@*
shared_name *
dtype0*"
_output_shapes
:@*
	container 
Ф
conv1d_2/kernel/AssignAssignconv1d_2/kernelconv1d_2/random_uniform*
T0*"
_class
loc:@conv1d_2/kernel*
validate_shape(*"
_output_shapes
:@*
use_locking(

conv1d_2/kernel/readIdentityconv1d_2/kernel*"
_output_shapes
:@*
T0*"
_class
loc:@conv1d_2/kernel
[
conv1d_2/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv1d_2/bias
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
­
conv1d_2/bias/AssignAssignconv1d_2/biasconv1d_2/Const*
use_locking(*
T0* 
_class
loc:@conv1d_2/bias*
validate_shape(*
_output_shapes
:@
t
conv1d_2/bias/readIdentityconv1d_2/bias*
_output_shapes
:@*
T0* 
_class
loc:@conv1d_2/bias
l
"conv1d_2/convolution/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
e
#conv1d_2/convolution/ExpandDims/dimConst*
_output_shapes
: *
value	B :*
dtype0
Ј
conv1d_2/convolution/ExpandDims
ExpandDimsconv1d_1/Relu#conv1d_2/convolution/ExpandDims/dim*0
_output_shapes
:џџџџџџџџџЌ*

Tdim0*
T0
g
%conv1d_2/convolution/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Љ
!conv1d_2/convolution/ExpandDims_1
ExpandDimsconv1d_2/kernel/read%conv1d_2/convolution/ExpandDims_1/dim*&
_output_shapes
:@*

Tdim0*
T0

conv1d_2/convolution/Conv2DConv2Dconv1d_2/convolution/ExpandDims!conv1d_2/convolution/ExpandDims_1*0
_output_shapes
:џџџџџџџџџЅ@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID

conv1d_2/convolution/SqueezeSqueezeconv1d_2/convolution/Conv2D*
T0*,
_output_shapes
:џџџџџџџџџЅ@*
squeeze_dims

k
conv1d_2/Reshape/shapeConst*!
valueB"      @   *
dtype0*
_output_shapes
:

conv1d_2/ReshapeReshapeconv1d_2/bias/readconv1d_2/Reshape/shape*"
_output_shapes
:@*
T0*
Tshape0
z
conv1d_2/addAddconv1d_2/convolution/Squeezeconv1d_2/Reshape*
T0*,
_output_shapes
:џџџџџџџџџЅ@
Z
conv1d_2/ReluReluconv1d_2/add*,
_output_shapes
:џџџџџџџџџЅ@*
T0
l
lstm_1/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"@      
^
lstm_1/random_uniform/minConst*
valueB
 *:ЭО*
dtype0*
_output_shapes
: 
^
lstm_1/random_uniform/maxConst*
valueB
 *:Э>*
dtype0*
_output_shapes
: 
І
#lstm_1/random_uniform/RandomUniformRandomUniformlstm_1/random_uniform/shape*
dtype0*
_output_shapes

:@*
seed2шЫ*
seedБџх)*
T0
w
lstm_1/random_uniform/subSublstm_1/random_uniform/maxlstm_1/random_uniform/min*
T0*
_output_shapes
: 

lstm_1/random_uniform/mulMul#lstm_1/random_uniform/RandomUniformlstm_1/random_uniform/sub*
_output_shapes

:@*
T0
{
lstm_1/random_uniformAddlstm_1/random_uniform/mullstm_1/random_uniform/min*
T0*
_output_shapes

:@

lstm_1/kernel
VariableV2*
dtype0*
_output_shapes

:@*
	container *
shape
:@*
shared_name 
И
lstm_1/kernel/AssignAssignlstm_1/kernellstm_1/random_uniform* 
_class
loc:@lstm_1/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
x
lstm_1/kernel/readIdentitylstm_1/kernel*
_output_shapes

:@*
T0* 
_class
loc:@lstm_1/kernel
Ж
%lstm_1/recurrent_kernel/initial_valueConst*Y
valuePBN"@Ё!Пv[C>gfмОAЭ>Ѕb>x1ЅО<gОшzлНнЈ:ЮЗОгЌПуE=>XН Оџk/?)Е>*
dtype0*
_output_shapes

:

lstm_1/recurrent_kernel
VariableV2*
shape
:*
shared_name *
dtype0*
_output_shapes

:*
	container 
ц
lstm_1/recurrent_kernel/AssignAssignlstm_1/recurrent_kernel%lstm_1/recurrent_kernel/initial_value*
T0**
_class 
loc:@lstm_1/recurrent_kernel*
validate_shape(*
_output_shapes

:*
use_locking(

lstm_1/recurrent_kernel/readIdentitylstm_1/recurrent_kernel*
T0**
_class 
loc:@lstm_1/recurrent_kernel*
_output_shapes

:
Y
lstm_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
[
lstm_1/Const_1Const*
valueB*  ?*
dtype0*
_output_shapes
:
[
lstm_1/Const_2Const*
valueB*    *
dtype0*
_output_shapes
:
T
lstm_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

lstm_1/concatConcatV2lstm_1/Constlstm_1/Const_1lstm_1/Const_2lstm_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
w
lstm_1/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
І
lstm_1/bias/AssignAssignlstm_1/biaslstm_1/concat*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@lstm_1/bias
n
lstm_1/bias/readIdentitylstm_1/bias*
T0*
_class
loc:@lstm_1/bias*
_output_shapes
:
k
lstm_1/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
m
lstm_1/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
m
lstm_1/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Њ
lstm_1/strided_sliceStridedSlicelstm_1/kernel/readlstm_1/strided_slice/stacklstm_1/strided_slice/stack_1lstm_1/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:@
m
lstm_1/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB"       
o
lstm_1/strided_slice_1/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
В
lstm_1/strided_slice_1StridedSlicelstm_1/kernel/readlstm_1/strided_slice_1/stacklstm_1/strided_slice_1/stack_1lstm_1/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:@
m
lstm_1/strided_slice_2/stackConst*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
valueB"       *
dtype0
o
lstm_1/strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
В
lstm_1/strided_slice_2StridedSlicelstm_1/kernel/readlstm_1/strided_slice_2/stacklstm_1/strided_slice_2/stack_1lstm_1/strided_slice_2/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:@*
T0*
Index0
m
lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
valueB"       *
dtype0
o
lstm_1/strided_slice_3/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
В
lstm_1/strided_slice_3StridedSlicelstm_1/kernel/readlstm_1/strided_slice_3/stacklstm_1/strided_slice_3/stack_1lstm_1/strided_slice_3/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:@
m
lstm_1/strided_slice_4/stackConst*
valueB"        *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_4/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
М
lstm_1/strided_slice_4StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_4/stacklstm_1/strided_slice_4/stack_1lstm_1/strided_slice_4/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:*
T0*
Index0
m
lstm_1/strided_slice_5/stackConst*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_5/stack_1Const*
_output_shapes
:*
valueB"       *
dtype0
o
lstm_1/strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
М
lstm_1/strided_slice_5StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_5/stacklstm_1/strided_slice_5/stack_1lstm_1/strided_slice_5/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:
m
lstm_1/strided_slice_6/stackConst*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_6/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
o
lstm_1/strided_slice_6/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
М
lstm_1/strided_slice_6StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_6/stacklstm_1/strided_slice_6/stack_1lstm_1/strided_slice_6/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:*
Index0*
T0
m
lstm_1/strided_slice_7/stackConst*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_7/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_7/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
М
lstm_1/strided_slice_7StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_7/stacklstm_1/strided_slice_7/stack_1lstm_1/strided_slice_7/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:
f
lstm_1/strided_slice_8/stackConst*
valueB: *
dtype0*
_output_shapes
:
h
lstm_1/strided_slice_8/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
h
lstm_1/strided_slice_8/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Ќ
lstm_1/strided_slice_8StridedSlicelstm_1/bias/readlstm_1/strided_slice_8/stacklstm_1/strided_slice_8/stack_1lstm_1/strided_slice_8/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0
f
lstm_1/strided_slice_9/stackConst*
valueB:*
dtype0*
_output_shapes
:
h
lstm_1/strided_slice_9/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
h
lstm_1/strided_slice_9/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Ќ
lstm_1/strided_slice_9StridedSlicelstm_1/bias/readlstm_1/strided_slice_9/stacklstm_1/strided_slice_9/stack_1lstm_1/strided_slice_9/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0
g
lstm_1/strided_slice_10/stackConst*
valueB:*
dtype0*
_output_shapes
:
i
lstm_1/strided_slice_10/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
i
lstm_1/strided_slice_10/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
А
lstm_1/strided_slice_10StridedSlicelstm_1/bias/readlstm_1/strided_slice_10/stacklstm_1/strided_slice_10/stack_1lstm_1/strided_slice_10/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0*
shrink_axis_mask 
g
lstm_1/strided_slice_11/stackConst*
dtype0*
_output_shapes
:*
valueB:
i
lstm_1/strided_slice_11/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
i
lstm_1/strided_slice_11/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
А
lstm_1/strided_slice_11StridedSlicelstm_1/bias/readlstm_1/strided_slice_11/stacklstm_1/strided_slice_11/stack_1lstm_1/strided_slice_11/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
Index0*
T0
d
lstm_1/zeros_like	ZerosLikeconv1d_2/Relu*
T0*,
_output_shapes
:џџџџџџџџџЅ@
m
lstm_1/Sum/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:


lstm_1/SumSumlstm_1/zeros_likelstm_1/Sum/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( *

Tidx0
`
lstm_1/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

lstm_1/ExpandDims
ExpandDims
lstm_1/Sumlstm_1/ExpandDims/dim*'
_output_shapes
:џџџџџџџџџ*

Tdim0*
T0
f
lstm_1/Tile/multiplesConst*
valueB"      *
dtype0*
_output_shapes
:

lstm_1/TileTilelstm_1/ExpandDimslstm_1/Tile/multiples*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
h
lstm_1/Tile_1/multiplesConst*
valueB"      *
dtype0*
_output_shapes
:

lstm_1/Tile_1Tilelstm_1/ExpandDimslstm_1/Tile_1/multiples*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
j
lstm_1/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:

lstm_1/transpose	Transposeconv1d_2/Relulstm_1/transpose/perm*
T0*,
_output_shapes
:Ѕџџџџџџџџџ@*
Tperm0
\
lstm_1/ShapeShapelstm_1/transpose*
_output_shapes
:*
T0*
out_type0
g
lstm_1/strided_slice_12/stackConst*
dtype0*
_output_shapes
:*
valueB: 
i
lstm_1/strided_slice_12/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
i
lstm_1/strided_slice_12/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ј
lstm_1/strided_slice_12StridedSlicelstm_1/Shapelstm_1/strided_slice_12/stacklstm_1/strided_slice_12/stack_1lstm_1/strided_slice_12/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
g
lstm_1/strided_slice_13/stackConst*
valueB: *
dtype0*
_output_shapes
:
i
lstm_1/strided_slice_13/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
i
lstm_1/strided_slice_13/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Н
lstm_1/strided_slice_13StridedSlicelstm_1/transposelstm_1/strided_slice_13/stacklstm_1/strided_slice_13/stack_1lstm_1/strided_slice_13/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *'
_output_shapes
:џџџџџџџџџ@*
Index0*
T0*
shrink_axis_mask

lstm_1/MatMulMatMullstm_1/strided_slice_13lstm_1/strided_slice*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ђ
lstm_1/MatMul_1MatMullstm_1/strided_slice_13lstm_1/strided_slice_1*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
Ђ
lstm_1/MatMul_2MatMullstm_1/strided_slice_13lstm_1/strided_slice_2*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ђ
lstm_1/MatMul_3MatMullstm_1/strided_slice_13lstm_1/strided_slice_3*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 

lstm_1/BiasAddBiasAddlstm_1/MatMullstm_1/strided_slice_8*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ

lstm_1/BiasAdd_1BiasAddlstm_1/MatMul_1lstm_1/strided_slice_9*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0

lstm_1/BiasAdd_2BiasAddlstm_1/MatMul_2lstm_1/strided_slice_10*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0

lstm_1/BiasAdd_3BiasAddlstm_1/MatMul_3lstm_1/strided_slice_11*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMul_4MatMullstm_1/Tilelstm_1/strided_slice_4*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
d

lstm_1/addAddlstm_1/BiasAddlstm_1/MatMul_4*
T0*'
_output_shapes
:џџџџџџџџџ
Q
lstm_1/mul/xConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
]

lstm_1/mulMullstm_1/mul/x
lstm_1/add*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/add_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *   ?
a
lstm_1/add_1Add
lstm_1/mullstm_1/add_1/y*'
_output_shapes
:џџџџџџџџџ*
T0
S
lstm_1/Const_3Const*
valueB
 *    *
dtype0*
_output_shapes
: 
S
lstm_1/Const_4Const*
_output_shapes
: *
valueB
 *  ?*
dtype0
w
lstm_1/clip_by_value/MinimumMinimumlstm_1/add_1lstm_1/Const_4*'
_output_shapes
:џџџџџџџџџ*
T0

lstm_1/clip_by_valueMaximumlstm_1/clip_by_value/Minimumlstm_1/Const_3*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMul_5MatMullstm_1/Tilelstm_1/strided_slice_5*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
h
lstm_1/add_2Addlstm_1/BiasAdd_1lstm_1/MatMul_5*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/mul_1/xConst*
_output_shapes
: *
valueB
 *ЭЬL>*
dtype0
c
lstm_1/mul_1Mullstm_1/mul_1/xlstm_1/add_2*'
_output_shapes
:џџџџџџџџџ*
T0
S
lstm_1/add_3/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
c
lstm_1/add_3Addlstm_1/mul_1lstm_1/add_3/y*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/Const_5Const*
valueB
 *    *
dtype0*
_output_shapes
: 
S
lstm_1/Const_6Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
y
lstm_1/clip_by_value_1/MinimumMinimumlstm_1/add_3lstm_1/Const_6*'
_output_shapes
:џџџџџџџџџ*
T0

lstm_1/clip_by_value_1Maximumlstm_1/clip_by_value_1/Minimumlstm_1/Const_5*
T0*'
_output_shapes
:џџџџџџџџџ
l
lstm_1/mul_2Mullstm_1/clip_by_value_1lstm_1/Tile_1*'
_output_shapes
:џџџџџџџџџ*
T0

lstm_1/MatMul_6MatMullstm_1/Tilelstm_1/strided_slice_6*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 
h
lstm_1/add_4Addlstm_1/BiasAdd_2lstm_1/MatMul_6*'
_output_shapes
:џџџџџџџџџ*
T0
S
lstm_1/TanhTanhlstm_1/add_4*'
_output_shapes
:џџџџџџџџџ*
T0
h
lstm_1/mul_3Mullstm_1/clip_by_valuelstm_1/Tanh*
T0*'
_output_shapes
:џџџџџџџџџ
a
lstm_1/add_5Addlstm_1/mul_2lstm_1/mul_3*'
_output_shapes
:џџџџџџџџџ*
T0

lstm_1/MatMul_7MatMullstm_1/Tilelstm_1/strided_slice_7*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
h
lstm_1/add_6Addlstm_1/BiasAdd_3lstm_1/MatMul_7*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/mul_4/xConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
c
lstm_1/mul_4Mullstm_1/mul_4/xlstm_1/add_6*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/add_7/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
c
lstm_1/add_7Addlstm_1/mul_4lstm_1/add_7/y*'
_output_shapes
:џџџџџџџџџ*
T0
S
lstm_1/Const_7Const*
_output_shapes
: *
valueB
 *    *
dtype0
S
lstm_1/Const_8Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
y
lstm_1/clip_by_value_2/MinimumMinimumlstm_1/add_7lstm_1/Const_8*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/clip_by_value_2Maximumlstm_1/clip_by_value_2/Minimumlstm_1/Const_7*'
_output_shapes
:џџџџџџџџџ*
T0
U
lstm_1/Tanh_1Tanhlstm_1/add_5*
T0*'
_output_shapes
:џџџџџџџџџ
l
lstm_1/mul_5Mullstm_1/clip_by_value_2lstm_1/Tanh_1*
T0*'
_output_shapes
:џџџџџџџџџ
ь
lstm_1/TensorArrayTensorArrayV3lstm_1/strided_slice_12*
dtype0*
_output_shapes

:: *
element_shape:*
clear_after_read(*
dynamic_size( *
identical_element_shapes(* 
tensor_array_name	output_ta
э
lstm_1/TensorArray_1TensorArrayV3lstm_1/strided_slice_12*
identical_element_shapes(*
tensor_array_name
input_ta*
dtype0*
_output_shapes

:: *
element_shape:*
clear_after_read(*
dynamic_size( 
o
lstm_1/TensorArrayUnstack/ShapeShapelstm_1/transpose*
T0*
out_type0*
_output_shapes
:
w
-lstm_1/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
y
/lstm_1/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/lstm_1/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ћ
'lstm_1/TensorArrayUnstack/strided_sliceStridedSlicelstm_1/TensorArrayUnstack/Shape-lstm_1/TensorArrayUnstack/strided_slice/stack/lstm_1/TensorArrayUnstack/strided_slice/stack_1/lstm_1/TensorArrayUnstack/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
g
%lstm_1/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
g
%lstm_1/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
а
lstm_1/TensorArrayUnstack/rangeRange%lstm_1/TensorArrayUnstack/range/start'lstm_1/TensorArrayUnstack/strided_slice%lstm_1/TensorArrayUnstack/range/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0

Alstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3lstm_1/TensorArray_1lstm_1/TensorArrayUnstack/rangelstm_1/transposelstm_1/TensorArray_1:1*
T0*#
_class
loc:@lstm_1/transpose*
_output_shapes
: 
M
lstm_1/timeConst*
value	B : *
dtype0*
_output_shapes
: 
b
lstm_1/while/maximum_iterationsConst*
value
B :Ѕ*
dtype0*
_output_shapes
: 
`
lstm_1/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
Ж
lstm_1/while/EnterEnterlstm_1/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: **

frame_namelstm_1/while/while_context
Ѕ
lstm_1/while/Enter_1Enterlstm_1/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: **

frame_namelstm_1/while/while_context
Ў
lstm_1/while/Enter_2Enterlstm_1/TensorArray:1*
parallel_iterations *
_output_shapes
: **

frame_namelstm_1/while/while_context*
T0*
is_constant( 
Ж
lstm_1/while/Enter_3Enterlstm_1/Tile*'
_output_shapes
:џџџџџџџџџ**

frame_namelstm_1/while/while_context*
T0*
is_constant( *
parallel_iterations 
И
lstm_1/while/Enter_4Enterlstm_1/Tile_1*'
_output_shapes
:џџџџџџџџџ**

frame_namelstm_1/while/while_context*
T0*
is_constant( *
parallel_iterations 
w
lstm_1/while/MergeMergelstm_1/while/Enterlstm_1/while/NextIteration*
T0*
N*
_output_shapes
: : 
}
lstm_1/while/Merge_1Mergelstm_1/while/Enter_1lstm_1/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
}
lstm_1/while/Merge_2Mergelstm_1/while/Enter_2lstm_1/while/NextIteration_2*
T0*
N*
_output_shapes
: : 

lstm_1/while/Merge_3Mergelstm_1/while/Enter_3lstm_1/while/NextIteration_3*
N*)
_output_shapes
:џџџџџџџџџ: *
T0

lstm_1/while/Merge_4Mergelstm_1/while/Enter_4lstm_1/while/NextIteration_4*
T0*
N*)
_output_shapes
:џџџџџџџџџ: 
g
lstm_1/while/LessLesslstm_1/while/Mergelstm_1/while/Less/Enter*
T0*
_output_shapes
: 
М
lstm_1/while/Less/EnterEnterlstm_1/while/maximum_iterations*
is_constant(*
parallel_iterations *
_output_shapes
: **

frame_namelstm_1/while/while_context*
T0
m
lstm_1/while/Less_1Lesslstm_1/while/Merge_1lstm_1/while/Less_1/Enter*
T0*
_output_shapes
: 
Ж
lstm_1/while/Less_1/EnterEnterlstm_1/strided_slice_12*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: **

frame_namelstm_1/while/while_context
e
lstm_1/while/LogicalAnd
LogicalAndlstm_1/while/Lesslstm_1/while/Less_1*
_output_shapes
: 
R
lstm_1/while/LoopCondLoopCondlstm_1/while/LogicalAnd*
_output_shapes
: 

lstm_1/while/SwitchSwitchlstm_1/while/Mergelstm_1/while/LoopCond*
T0*%
_class
loc:@lstm_1/while/Merge*
_output_shapes
: : 

lstm_1/while/Switch_1Switchlstm_1/while/Merge_1lstm_1/while/LoopCond*
T0*'
_class
loc:@lstm_1/while/Merge_1*
_output_shapes
: : 

lstm_1/while/Switch_2Switchlstm_1/while/Merge_2lstm_1/while/LoopCond*
T0*'
_class
loc:@lstm_1/while/Merge_2*
_output_shapes
: : 
К
lstm_1/while/Switch_3Switchlstm_1/while/Merge_3lstm_1/while/LoopCond*'
_class
loc:@lstm_1/while/Merge_3*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*
T0
К
lstm_1/while/Switch_4Switchlstm_1/while/Merge_4lstm_1/while/LoopCond*
T0*'
_class
loc:@lstm_1/while/Merge_4*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
Y
lstm_1/while/IdentityIdentitylstm_1/while/Switch:1*
_output_shapes
: *
T0
]
lstm_1/while/Identity_1Identitylstm_1/while/Switch_1:1*
_output_shapes
: *
T0
]
lstm_1/while/Identity_2Identitylstm_1/while/Switch_2:1*
T0*
_output_shapes
: 
n
lstm_1/while/Identity_3Identitylstm_1/while/Switch_3:1*
T0*'
_output_shapes
:џџџџџџџџџ
n
lstm_1/while/Identity_4Identitylstm_1/while/Switch_4:1*
T0*'
_output_shapes
:џџџџџџџџџ
l
lstm_1/while/add/yConst^lstm_1/while/Identity*
dtype0*
_output_shapes
: *
value	B :
c
lstm_1/while/addAddlstm_1/while/Identitylstm_1/while/add/y*
T0*
_output_shapes
: 
а
lstm_1/while/TensorArrayReadV3TensorArrayReadV3$lstm_1/while/TensorArrayReadV3/Enterlstm_1/while/Identity_1&lstm_1/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:џџџџџџџџџ@
Т
$lstm_1/while/TensorArrayReadV3/EnterEnterlstm_1/TensorArray_1*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*
is_constant(*
parallel_iterations 
э
&lstm_1/while/TensorArrayReadV3/Enter_1EnterAlstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: **

frame_namelstm_1/while/while_context
А
lstm_1/while/MatMulMatMullstm_1/while/TensorArrayReadV3lstm_1/while/MatMul/Enter*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Л
lstm_1/while/MatMul/EnterEnterlstm_1/strided_slice*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:@**

frame_namelstm_1/while/while_context
Д
lstm_1/while/MatMul_1MatMullstm_1/while/TensorArrayReadV3lstm_1/while/MatMul_1/Enter*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
П
lstm_1/while/MatMul_1/EnterEnterlstm_1/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:@**

frame_namelstm_1/while/while_context
Д
lstm_1/while/MatMul_2MatMullstm_1/while/TensorArrayReadV3lstm_1/while/MatMul_2/Enter*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 
П
lstm_1/while/MatMul_2/EnterEnterlstm_1/strided_slice_2*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:@**

frame_namelstm_1/while/while_context
Д
lstm_1/while/MatMul_3MatMullstm_1/while/TensorArrayReadV3lstm_1/while/MatMul_3/Enter*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
П
lstm_1/while/MatMul_3/EnterEnterlstm_1/strided_slice_3*
_output_shapes

:@**

frame_namelstm_1/while/while_context*
T0*
is_constant(*
parallel_iterations 

lstm_1/while/BiasAddBiasAddlstm_1/while/MatMullstm_1/while/BiasAdd/Enter*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
К
lstm_1/while/BiasAdd/EnterEnterlstm_1/strided_slice_8*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelstm_1/while/while_context

lstm_1/while/BiasAdd_1BiasAddlstm_1/while/MatMul_1lstm_1/while/BiasAdd_1/Enter*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
М
lstm_1/while/BiasAdd_1/EnterEnterlstm_1/strided_slice_9*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0

lstm_1/while/BiasAdd_2BiasAddlstm_1/while/MatMul_2lstm_1/while/BiasAdd_2/Enter*'
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
Н
lstm_1/while/BiasAdd_2/EnterEnterlstm_1/strided_slice_10*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelstm_1/while/while_context

lstm_1/while/BiasAdd_3BiasAddlstm_1/while/MatMul_3lstm_1/while/BiasAdd_3/Enter*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
Н
lstm_1/while/BiasAdd_3/EnterEnterlstm_1/strided_slice_11*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*
is_constant(*
parallel_iterations 
­
lstm_1/while/MatMul_4MatMullstm_1/while/Identity_3lstm_1/while/MatMul_4/Enter*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
П
lstm_1/while/MatMul_4/EnterEnterlstm_1/strided_slice_4*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context*
T0*
is_constant(
x
lstm_1/while/add_1Addlstm_1/while/BiasAddlstm_1/while/MatMul_4*'
_output_shapes
:џџџџџџџџџ*
T0
o
lstm_1/while/mul/xConst^lstm_1/while/Identity*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
q
lstm_1/while/mulMullstm_1/while/mul/xlstm_1/while/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/add_2/yConst^lstm_1/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *   ?
s
lstm_1/while/add_2Addlstm_1/while/mullstm_1/while/add_2/y*
T0*'
_output_shapes
:џџџџџџџџџ
o
lstm_1/while/ConstConst^lstm_1/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
q
lstm_1/while/Const_1Const^lstm_1/while/Identity*
_output_shapes
: *
valueB
 *  ?*
dtype0

"lstm_1/while/clip_by_value/MinimumMinimumlstm_1/while/add_2lstm_1/while/Const_1*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/while/clip_by_valueMaximum"lstm_1/while/clip_by_value/Minimumlstm_1/while/Const*
T0*'
_output_shapes
:џџџџџџџџџ
­
lstm_1/while/MatMul_5MatMullstm_1/while/Identity_3lstm_1/while/MatMul_5/Enter*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 
П
lstm_1/while/MatMul_5/EnterEnterlstm_1/strided_slice_5*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context
z
lstm_1/while/add_3Addlstm_1/while/BiasAdd_1lstm_1/while/MatMul_5*'
_output_shapes
:џџџџџџџџџ*
T0
q
lstm_1/while/mul_1/xConst^lstm_1/while/Identity*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
u
lstm_1/while/mul_1Mullstm_1/while/mul_1/xlstm_1/while/add_3*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/add_4/yConst^lstm_1/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 
u
lstm_1/while/add_4Addlstm_1/while/mul_1lstm_1/while/add_4/y*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/Const_2Const^lstm_1/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
q
lstm_1/while/Const_3Const^lstm_1/while/Identity*
_output_shapes
: *
valueB
 *  ?*
dtype0

$lstm_1/while/clip_by_value_1/MinimumMinimumlstm_1/while/add_4lstm_1/while/Const_3*'
_output_shapes
:џџџџџџџџџ*
T0

lstm_1/while/clip_by_value_1Maximum$lstm_1/while/clip_by_value_1/Minimumlstm_1/while/Const_2*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/while/mul_2Mullstm_1/while/clip_by_value_1lstm_1/while/Identity_4*
T0*'
_output_shapes
:џџџџџџџџџ
­
lstm_1/while/MatMul_6MatMullstm_1/while/Identity_3lstm_1/while/MatMul_6/Enter*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
П
lstm_1/while/MatMul_6/EnterEnterlstm_1/strided_slice_6*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context
z
lstm_1/while/add_5Addlstm_1/while/BiasAdd_2lstm_1/while/MatMul_6*'
_output_shapes
:џџџџџџџџџ*
T0
_
lstm_1/while/TanhTanhlstm_1/while/add_5*
T0*'
_output_shapes
:џџџџџџџџџ
z
lstm_1/while/mul_3Mullstm_1/while/clip_by_valuelstm_1/while/Tanh*'
_output_shapes
:џџџџџџџџџ*
T0
s
lstm_1/while/add_6Addlstm_1/while/mul_2lstm_1/while/mul_3*
T0*'
_output_shapes
:џџџџџџџџџ
­
lstm_1/while/MatMul_7MatMullstm_1/while/Identity_3lstm_1/while/MatMul_7/Enter*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
П
lstm_1/while/MatMul_7/EnterEnterlstm_1/strided_slice_7*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context
z
lstm_1/while/add_7Addlstm_1/while/BiasAdd_3lstm_1/while/MatMul_7*'
_output_shapes
:џџџџџџџџџ*
T0
q
lstm_1/while/mul_4/xConst^lstm_1/while/Identity*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
u
lstm_1/while/mul_4Mullstm_1/while/mul_4/xlstm_1/while/add_7*'
_output_shapes
:џџџџџџџџџ*
T0
q
lstm_1/while/add_8/yConst^lstm_1/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 
u
lstm_1/while/add_8Addlstm_1/while/mul_4lstm_1/while/add_8/y*'
_output_shapes
:џџџџџџџџџ*
T0
q
lstm_1/while/Const_4Const^lstm_1/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
q
lstm_1/while/Const_5Const^lstm_1/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$lstm_1/while/clip_by_value_2/MinimumMinimumlstm_1/while/add_8lstm_1/while/Const_5*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/while/clip_by_value_2Maximum$lstm_1/while/clip_by_value_2/Minimumlstm_1/while/Const_4*
T0*'
_output_shapes
:џџџџџџџџџ
a
lstm_1/while/Tanh_1Tanhlstm_1/while/add_6*'
_output_shapes
:џџџџџџџџџ*
T0
~
lstm_1/while/mul_5Mullstm_1/while/clip_by_value_2lstm_1/while/Tanh_1*
T0*'
_output_shapes
:џџџџџџџџџ

0lstm_1/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV36lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/Enterlstm_1/while/Identity_1lstm_1/while/mul_5lstm_1/while/Identity_2*
T0*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
: 
љ
6lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterlstm_1/TensorArray*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations 
n
lstm_1/while/add_9/yConst^lstm_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
i
lstm_1/while/add_9Addlstm_1/while/Identity_1lstm_1/while/add_9/y*
T0*
_output_shapes
: 
^
lstm_1/while/NextIterationNextIterationlstm_1/while/add*
T0*
_output_shapes
: 
b
lstm_1/while/NextIteration_1NextIterationlstm_1/while/add_9*
T0*
_output_shapes
: 

lstm_1/while/NextIteration_2NextIteration0lstm_1/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
s
lstm_1/while/NextIteration_3NextIterationlstm_1/while/mul_5*'
_output_shapes
:џџџџџџџџџ*
T0
s
lstm_1/while/NextIteration_4NextIterationlstm_1/while/add_6*
T0*'
_output_shapes
:џџџџџџџџџ
O
lstm_1/while/ExitExitlstm_1/while/Switch*
T0*
_output_shapes
: 
S
lstm_1/while/Exit_1Exitlstm_1/while/Switch_1*
_output_shapes
: *
T0
S
lstm_1/while/Exit_2Exitlstm_1/while/Switch_2*
T0*
_output_shapes
: 
d
lstm_1/while/Exit_3Exitlstm_1/while/Switch_3*'
_output_shapes
:џџџџџџџџџ*
T0
d
lstm_1/while/Exit_4Exitlstm_1/while/Switch_4*'
_output_shapes
:џџџџџџџџџ*
T0
І
)lstm_1/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3lstm_1/TensorArraylstm_1/while/Exit_2*%
_class
loc:@lstm_1/TensorArray*
_output_shapes
: 

#lstm_1/TensorArrayStack/range/startConst*%
_class
loc:@lstm_1/TensorArray*
value	B : *
dtype0*
_output_shapes
: 

#lstm_1/TensorArrayStack/range/deltaConst*%
_class
loc:@lstm_1/TensorArray*
value	B :*
dtype0*
_output_shapes
: 
ѓ
lstm_1/TensorArrayStack/rangeRange#lstm_1/TensorArrayStack/range/start)lstm_1/TensorArrayStack/TensorArraySizeV3#lstm_1/TensorArrayStack/range/delta*

Tidx0*%
_class
loc:@lstm_1/TensorArray*#
_output_shapes
:џџџџџџџџџ

+lstm_1/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3lstm_1/TensorArraylstm_1/TensorArrayStack/rangelstm_1/while/Exit_2*$
element_shape:џџџџџџџџџ*%
_class
loc:@lstm_1/TensorArray*
dtype0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
N
lstm_1/sub/yConst*
dtype0*
_output_shapes
: *
value	B :
U

lstm_1/subSublstm_1/while/Exit_1lstm_1/sub/y*
T0*
_output_shapes
: 

lstm_1/TensorArrayReadV3TensorArrayReadV3lstm_1/TensorArray
lstm_1/sublstm_1/while/Exit_2*
dtype0*'
_output_shapes
:џџџџџџџџџ
l
lstm_1/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:
Б
lstm_1/transpose_1	Transpose+lstm_1/TensorArrayStack/TensorArrayGatherV3lstm_1/transpose_1/perm*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
Tperm0
f
$dropout_1/keras_learning_phase/inputConst*
value	B
 Z *
dtype0
*
_output_shapes
: 

dropout_1/keras_learning_phasePlaceholderWithDefault$dropout_1/keras_learning_phase/input*
shape: *
dtype0
*
_output_shapes
: 

dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes
: : *
T0

]
dropout_1/cond/switch_tIdentitydropout_1/cond/Switch:1*
_output_shapes
: *
T0

[
dropout_1/cond/switch_fIdentitydropout_1/cond/Switch*
T0
*
_output_shapes
: 
c
dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
: 
s
dropout_1/cond/mul/yConst^dropout_1/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 

dropout_1/cond/mulMuldropout_1/cond/mul/Switch:1dropout_1/cond/mul/y*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0
е
dropout_1/cond/mul/SwitchSwitchlstm_1/transpose_1dropout_1/cond/pred_id*
T0*%
_class
loc:@lstm_1/transpose_1*T
_output_shapesB
@:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ

 dropout_1/cond/dropout/keep_probConst^dropout_1/cond/switch_t*
valueB
 *fff?*
dtype0*
_output_shapes
: 
n
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
T0*
out_type0*
_output_shapes
:

)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
Ь
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
T0*
dtype0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
seed2ФГp*
seedБџх)
Ї
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Я
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0
С
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0
Љ
dropout_1/cond/dropout/addAdd dropout_1/cond/dropout/keep_prob%dropout_1/cond/dropout/random_uniform*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0

dropout_1/cond/dropout/FloorFloordropout_1/cond/dropout/add*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0

dropout_1/cond/dropout/divRealDivdropout_1/cond/mul dropout_1/cond/dropout/keep_prob*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0

dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/divdropout_1/cond/dropout/Floor*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
г
dropout_1/cond/Switch_1Switchlstm_1/transpose_1dropout_1/cond/pred_id*
T0*%
_class
loc:@lstm_1/transpose_1*T
_output_shapesB
@:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ

dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*
N*6
_output_shapes$
":џџџџџџџџџџџџџџџџџџ: *
T0
c
flatten_1/ShapeShapedropout_1/cond/Merge*
_output_shapes
:*
T0*
out_type0
g
flatten_1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Џ
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
T0*
Index0
Y
flatten_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
~
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
\
flatten_1/stack/0Const*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ
t
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*
T0*

axis *
N*
_output_shapes
:

flatten_1/ReshapeReshapedropout_1/cond/Mergeflatten_1/stack*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
m
dense_1/random_uniform/shapeConst*
valueB"J     *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *hъН
_
dense_1/random_uniform/maxConst*
valueB
 *hъ=*
dtype0*
_output_shapes
: 
Њ
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
seedБџх)*
T0*
dtype0* 
_output_shapes
:
Ъ*
seed2ЁЅХ
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 

dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0* 
_output_shapes
:
Ъ

dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0* 
_output_shapes
:
Ъ

dense_1/kernel
VariableV2*
dtype0* 
_output_shapes
:
Ъ*
	container *
shape:
Ъ*
shared_name 
О
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
validate_shape(* 
_output_shapes
:
Ъ*
use_locking(*
T0*!
_class
loc:@dense_1/kernel
}
dense_1/kernel/readIdentitydense_1/kernel* 
_output_shapes
:
Ъ*
T0*!
_class
loc:@dense_1/kernel
\
dense_1/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
z
dense_1/bias
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
Њ
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@dense_1/bias
r
dense_1/bias/readIdentitydense_1/bias*
_output_shapes	
:*
T0*
_class
loc:@dense_1/bias

dense_1/MatMulMatMulflatten_1/Reshapedense_1/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
X
dense_1/ReluReludense_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes
: : 
]
dropout_2/cond/switch_tIdentitydropout_2/cond/Switch:1*
T0
*
_output_shapes
: 
[
dropout_2/cond/switch_fIdentitydropout_2/cond/Switch*
T0
*
_output_shapes
: 
c
dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
: 
s
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 

dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*(
_output_shapes
:џџџџџџџџџ*
T0
Б
dropout_2/cond/mul/SwitchSwitchdense_1/Reludropout_2/cond/pred_id*
T0*
_class
loc:@dense_1/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ

 dropout_2/cond/dropout/keep_probConst^dropout_2/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
n
dropout_2/cond/dropout/ShapeShapedropout_2/cond/mul*
T0*
out_type0*
_output_shapes
:

)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

)dropout_2/cond/dropout/random_uniform/maxConst^dropout_2/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Р
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
T0*
dtype0*(
_output_shapes
:џџџџџџџџџ*
seed2,*
seedБџх)
Ї
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
У
)dropout_2/cond/dropout/random_uniform/mulMul3dropout_2/cond/dropout/random_uniform/RandomUniform)dropout_2/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:џџџџџџџџџ
Е
%dropout_2/cond/dropout/random_uniformAdd)dropout_2/cond/dropout/random_uniform/mul)dropout_2/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_2/cond/dropout/addAdd dropout_2/cond/dropout/keep_prob%dropout_2/cond/dropout/random_uniform*(
_output_shapes
:џџџџџџџџџ*
T0
t
dropout_2/cond/dropout/FloorFloordropout_2/cond/dropout/add*(
_output_shapes
:џџџџџџџџџ*
T0

dropout_2/cond/dropout/divRealDivdropout_2/cond/mul dropout_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_2/cond/dropout/mulMuldropout_2/cond/dropout/divdropout_2/cond/dropout/Floor*
T0*(
_output_shapes
:џџџџџџџџџ
Џ
dropout_2/cond/Switch_1Switchdense_1/Reludropout_2/cond/pred_id*
T0*
_class
loc:@dense_1/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ

dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul*
N**
_output_shapes
:џџџџџџџџџ: *
T0
m
dense_2/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
_output_shapes
: *
valueB
 *qФО*
dtype0
_
dense_2/random_uniform/maxConst*
valueB
 *qФ>*
dtype0*
_output_shapes
: 
Њ
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
dtype0* 
_output_shapes
:
*
seed2ЛЅЎ*
seedБџх)*
T0
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
_output_shapes
: *
T0

dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub* 
_output_shapes
:
*
T0

dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0* 
_output_shapes
:


dense_2/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
	container *
shape:
*
shared_name 
О
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(* 
_output_shapes
:

}
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel* 
_output_shapes
:

\
dense_2/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
z
dense_2/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Њ
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(
r
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes	
:

dense_2/MatMulMatMuldropout_2/cond/Mergedense_2/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ*
T0
X
dense_2/ReluReludense_2/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_3/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes
: : 
]
dropout_3/cond/switch_tIdentitydropout_3/cond/Switch:1*
T0
*
_output_shapes
: 
[
dropout_3/cond/switch_fIdentitydropout_3/cond/Switch*
_output_shapes
: *
T0

c
dropout_3/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
: 
s
dropout_3/cond/mul/yConst^dropout_3/cond/switch_t*
_output_shapes
: *
valueB
 *  ?*
dtype0

dropout_3/cond/mulMuldropout_3/cond/mul/Switch:1dropout_3/cond/mul/y*(
_output_shapes
:џџџџџџџџџ*
T0
Б
dropout_3/cond/mul/SwitchSwitchdense_2/Reludropout_3/cond/pred_id*
T0*
_class
loc:@dense_2/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ

 dropout_3/cond/dropout/keep_probConst^dropout_3/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
n
dropout_3/cond/dropout/ShapeShapedropout_3/cond/mul*
T0*
out_type0*
_output_shapes
:

)dropout_3/cond/dropout/random_uniform/minConst^dropout_3/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

)dropout_3/cond/dropout/random_uniform/maxConst^dropout_3/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
С
3dropout_3/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_3/cond/dropout/Shape*
T0*
dtype0*(
_output_shapes
:џџџџџџџџџ*
seed2зЌй*
seedБџх)
Ї
)dropout_3/cond/dropout/random_uniform/subSub)dropout_3/cond/dropout/random_uniform/max)dropout_3/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
У
)dropout_3/cond/dropout/random_uniform/mulMul3dropout_3/cond/dropout/random_uniform/RandomUniform)dropout_3/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:џџџџџџџџџ
Е
%dropout_3/cond/dropout/random_uniformAdd)dropout_3/cond/dropout/random_uniform/mul)dropout_3/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_3/cond/dropout/addAdd dropout_3/cond/dropout/keep_prob%dropout_3/cond/dropout/random_uniform*(
_output_shapes
:џџџџџџџџџ*
T0
t
dropout_3/cond/dropout/FloorFloordropout_3/cond/dropout/add*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_3/cond/dropout/divRealDivdropout_3/cond/mul dropout_3/cond/dropout/keep_prob*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_3/cond/dropout/mulMuldropout_3/cond/dropout/divdropout_3/cond/dropout/Floor*
T0*(
_output_shapes
:џџџџџџџџџ
Џ
dropout_3/cond/Switch_1Switchdense_2/Reludropout_3/cond/pred_id*
T0*
_class
loc:@dense_2/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ

dropout_3/cond/MergeMergedropout_3/cond/Switch_1dropout_3/cond/dropout/mul*
T0*
N**
_output_shapes
:џџџџџџџџџ: 
m
dense_3/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
_
dense_3/random_uniform/minConst*
valueB
 *ЃЎXО*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
_output_shapes
: *
valueB
 *ЃЎX>*
dtype0
Ј
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
dtype0*
_output_shapes
:	*
seed2Љ*
seedБџх)*
T0
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
T0*
_output_shapes
: 

dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
T0*
_output_shapes
:	

dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
_output_shapes
:	*
T0

dense_3/kernel
VariableV2*
_output_shapes
:	*
	container *
shape:	*
shared_name *
dtype0
Н
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0
|
dense_3/kernel/readIdentitydense_3/kernel*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes
:	
Z
dense_3/ConstConst*
_output_shapes
:*
valueB*    *
dtype0
x
dense_3/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Љ
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:
q
dense_3/bias/readIdentitydense_3/bias*
_output_shapes
:*
T0*
_class
loc:@dense_3/bias

dense_3/MatMulMatMuldropout_3/cond/Mergedense_3/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
]
dense_3/SoftmaxSoftmaxdense_3/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
_
Adam/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
s
Adam/iterations
VariableV2*
shared_name *
dtype0	*
_output_shapes
: *
	container *
shape: 
О
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*
use_locking(*
T0	*"
_class
loc:@Adam/iterations*
validate_shape(*
_output_shapes
: 
v
Adam/iterations/readIdentityAdam/iterations*
T0	*"
_class
loc:@Adam/iterations*
_output_shapes
: 
Z
Adam/lr/initial_valueConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
k
Adam/lr
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 

Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
use_locking(*
T0*
_class
loc:@Adam/lr*
validate_shape(*
_output_shapes
: 
^
Adam/lr/readIdentityAdam/lr*
_output_shapes
: *
T0*
_class
loc:@Adam/lr
^
Adam/beta_1/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
o
Adam/beta_1
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
Ў
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/beta_1
j
Adam/beta_1/readIdentityAdam/beta_1*
T0*
_class
loc:@Adam/beta_1*
_output_shapes
: 
^
Adam/beta_2/initial_valueConst*
valueB
 *wО?*
dtype0*
_output_shapes
: 
o
Adam/beta_2
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Ў
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
use_locking(*
T0*
_class
loc:@Adam/beta_2*
validate_shape(*
_output_shapes
: 
j
Adam/beta_2/readIdentityAdam/beta_2*
T0*
_class
loc:@Adam/beta_2*
_output_shapes
: 
]
Adam/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n

Adam/decay
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
Њ
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
use_locking(*
T0*
_class
loc:@Adam/decay*
validate_shape(*
_output_shapes
: 
g
Adam/decay/readIdentity
Adam/decay*
_class
loc:@Adam/decay*
_output_shapes
: *
T0

dense_3_targetPlaceholder*%
shape:џџџџџџџџџџџџџџџџџџ*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
q
dense_3_sample_weightsPlaceholder*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ*
dtype0
r
'loss/dense_3_loss/Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ
Ѕ
loss/dense_3_loss/SumSumdense_3/Softmax'loss/dense_3_loss/Sum/reduction_indices*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0
~
loss/dense_3_loss/truedivRealDivdense_3/Softmaxloss/dense_3_loss/Sum*
T0*'
_output_shapes
:џџџџџџџџџ
\
loss/dense_3_loss/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *Пж3
\
loss/dense_3_loss/sub/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
o
loss/dense_3_loss/subSubloss/dense_3_loss/sub/xloss/dense_3_loss/Const*
T0*
_output_shapes
: 

'loss/dense_3_loss/clip_by_value/MinimumMinimumloss/dense_3_loss/truedivloss/dense_3_loss/sub*
T0*'
_output_shapes
:џџџџџџџџџ

loss/dense_3_loss/clip_by_valueMaximum'loss/dense_3_loss/clip_by_value/Minimumloss/dense_3_loss/Const*
T0*'
_output_shapes
:џџџџџџџџџ
o
loss/dense_3_loss/LogLogloss/dense_3_loss/clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџ
u
loss/dense_3_loss/mulMuldense_3_targetloss/dense_3_loss/Log*
T0*'
_output_shapes
:џџџџџџџџџ
t
)loss/dense_3_loss/Sum_1/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ћ
loss/dense_3_loss/Sum_1Sumloss/dense_3_loss/mul)loss/dense_3_loss/Sum_1/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:џџџџџџџџџ
c
loss/dense_3_loss/NegNegloss/dense_3_loss/Sum_1*
T0*#
_output_shapes
:џџџџџџџџџ
k
(loss/dense_3_loss/Mean/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
Њ
loss/dense_3_loss/MeanMeanloss/dense_3_loss/Neg(loss/dense_3_loss/Mean/reduction_indices*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( *

Tidx0*
T0
|
loss/dense_3_loss/mul_1Mulloss/dense_3_loss/Meandense_3_sample_weights*
T0*#
_output_shapes
:џџџџџџџџџ
a
loss/dense_3_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss/dense_3_loss/NotEqualNotEqualdense_3_sample_weightsloss/dense_3_loss/NotEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ

loss/dense_3_loss/CastCastloss/dense_3_loss/NotEqual*
Truncate( *#
_output_shapes
:џџџџџџџџџ*

DstT0*

SrcT0

c
loss/dense_3_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss/dense_3_loss/Mean_1Meanloss/dense_3_loss/Castloss/dense_3_loss/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

loss/dense_3_loss/truediv_1RealDivloss/dense_3_loss/mul_1loss/dense_3_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ*
T0
c
loss/dense_3_loss/Const_2Const*
valueB: *
dtype0*
_output_shapes
:

loss/dense_3_loss/Mean_2Meanloss/dense_3_loss/truediv_1loss/dense_3_loss/Const_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
O

loss/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
V
loss/mulMul
loss/mul/xloss/dense_3_loss/Mean_2*
T0*
_output_shapes
: 
g
metrics/acc/ArgMax/dimensionConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

metrics/acc/ArgMaxArgMaxdense_3_targetmetrics/acc/ArgMax/dimension*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0*
output_type0	
i
metrics/acc/ArgMax_1/dimensionConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

metrics/acc/ArgMax_1ArgMaxdense_3/Softmaxmetrics/acc/ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:џџџџџџџџџ*

Tidx0
r
metrics/acc/EqualEqualmetrics/acc/ArgMaxmetrics/acc/ArgMax_1*
T0	*#
_output_shapes
:џџџџџџџџџ
x
metrics/acc/CastCastmetrics/acc/Equal*

SrcT0
*
Truncate( *#
_output_shapes
:џџџџџџџџџ*

DstT0
[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
{
metrics/acc/MeanMeanmetrics/acc/Castmetrics/acc/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
}
training/Adam/gradients/ShapeConst*
_class
loc:@loss/mul*
valueB *
dtype0*
_output_shapes
: 

!training/Adam/gradients/grad_ys_0Const*
_class
loc:@loss/mul*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ж
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
_class
loc:@loss/mul*

index_type0*
_output_shapes
: *
T0

training/Adam/gradients/f_countConst*&
_class
loc:@lstm_1/while/Exit_2*
value	B : *
dtype0*
_output_shapes
: 
ю
!training/Adam/gradients/f_count_1Entertraining/Adam/gradients/f_count*
is_constant( *
_output_shapes
: **

frame_namelstm_1/while/while_context*
T0*&
_class
loc:@lstm_1/while/Exit_2*
parallel_iterations 
Ф
training/Adam/gradients/MergeMerge!training/Adam/gradients/f_count_1%training/Adam/gradients/NextIteration*
T0*&
_class
loc:@lstm_1/while/Exit_2*
N*
_output_shapes
: : 
Љ
training/Adam/gradients/SwitchSwitchtraining/Adam/gradients/Mergelstm_1/while/LoopCond*
T0*&
_class
loc:@lstm_1/while/Exit_2*
_output_shapes
: : 

training/Adam/gradients/Add/yConst^lstm_1/while/Identity*&
_class
loc:@lstm_1/while/Exit_2*
value	B :*
dtype0*
_output_shapes
: 
Ќ
training/Adam/gradients/AddAdd training/Adam/gradients/Switch:1training/Adam/gradients/Add/y*
T0*&
_class
loc:@lstm_1/while/Exit_2*
_output_shapes
: 

%training/Adam/gradients/NextIterationNextIterationtraining/Adam/gradients/AddH^training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPushV2H^training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPushV2l^training/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2R^training/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPushV2_1R^training/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPushV2R^training/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPushV2_1R^training/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPushV2R^training/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPushV2_1R^training/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPushV2_1R^training/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPushV2_1R^training/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPushV2b^training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPushV2V^training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPushV2d^training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPushV2X^training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPushV2\^training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPushV2S^training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPushV2d^training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPushV2X^training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPushV2\^training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPushV2S^training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPushV2Z^training/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPushV2Q^training/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPushV2R^training/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPushV2@^training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPushV2R^training/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPushV2_1@^training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPushV2B^training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPushV2R^training/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPushV2_1@^training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPushV2B^training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPushV2R^training/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPushV2@^training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPushV2R^training/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPushV2_1@^training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPushV2B^training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPushV2P^training/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPushV2>^training/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPushV2*
_output_shapes
: *
T0*&
_class
loc:@lstm_1/while/Exit_2

!training/Adam/gradients/f_count_2Exittraining/Adam/gradients/Switch*
_output_shapes
: *
T0*&
_class
loc:@lstm_1/while/Exit_2

training/Adam/gradients/b_countConst*&
_class
loc:@lstm_1/while/Exit_2*
value	B :*
dtype0*
_output_shapes
: 

!training/Adam/gradients/b_count_1Enter!training/Adam/gradients/f_count_2*
is_constant( *
_output_shapes
: *B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*&
_class
loc:@lstm_1/while/Exit_2*
parallel_iterations 
Ш
training/Adam/gradients/Merge_1Merge!training/Adam/gradients/b_count_1'training/Adam/gradients/NextIteration_1*
T0*&
_class
loc:@lstm_1/while/Exit_2*
N*
_output_shapes
: : 
Ъ
$training/Adam/gradients/GreaterEqualGreaterEqualtraining/Adam/gradients/Merge_1*training/Adam/gradients/GreaterEqual/Enter*
T0*&
_class
loc:@lstm_1/while/Exit_2*
_output_shapes
: 

*training/Adam/gradients/GreaterEqual/EnterEntertraining/Adam/gradients/b_count*
is_constant(*
_output_shapes
: *B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*&
_class
loc:@lstm_1/while/Exit_2*
parallel_iterations 

!training/Adam/gradients/b_count_2LoopCond$training/Adam/gradients/GreaterEqual*
_output_shapes
: *&
_class
loc:@lstm_1/while/Exit_2
Й
 training/Adam/gradients/Switch_1Switchtraining/Adam/gradients/Merge_1!training/Adam/gradients/b_count_2*
T0*&
_class
loc:@lstm_1/while/Exit_2*
_output_shapes
: : 
Л
training/Adam/gradients/SubSub"training/Adam/gradients/Switch_1:1*training/Adam/gradients/GreaterEqual/Enter*
T0*&
_class
loc:@lstm_1/while/Exit_2*
_output_shapes
: 

'training/Adam/gradients/NextIteration_1NextIterationtraining/Adam/gradients/Subg^training/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
_output_shapes
: *
T0*&
_class
loc:@lstm_1/while/Exit_2

!training/Adam/gradients/b_count_3Exit training/Adam/gradients/Switch_1*
T0*&
_class
loc:@lstm_1/while/Exit_2*
_output_shapes
: 
І
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/dense_3_loss/Mean_2*
T0*
_class
loc:@loss/mul*
_output_shapes
: 

+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
К
Ctraining/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Reshape/shapeConst*+
_class!
loc:@loss/dense_3_loss/Mean_2*
valueB:*
dtype0*
_output_shapes
:

=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Ctraining/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Reshape/shape*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*
Tshape0
У
;training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/ShapeShapeloss/dense_3_loss/truediv_1*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*
out_type0*
_output_shapes
:
Ћ
:training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/TileTile=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Reshape;training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Shape*#
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2
Х
=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Shape_1Shapeloss/dense_3_loss/truediv_1*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*
out_type0*
_output_shapes
:
­
=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Shape_2Const*
dtype0*
_output_shapes
: *+
_class!
loc:@loss/dense_3_loss/Mean_2*
valueB 
В
;training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/ConstConst*+
_class!
loc:@loss/dense_3_loss/Mean_2*
valueB: *
dtype0*
_output_shapes
:
Љ
:training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/ProdProd=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Shape_1;training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Const*
	keep_dims( *

Tidx0*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*
_output_shapes
: 
Д
=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Const_1Const*+
_class!
loc:@loss/dense_3_loss/Mean_2*
valueB: *
dtype0*
_output_shapes
:
­
<training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Prod_1Prod=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Shape_2=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Const_1*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*
_output_shapes
: *
	keep_dims( *

Tidx0
Ў
?training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Maximum/yConst*+
_class!
loc:@loss/dense_3_loss/Mean_2*
value	B :*
dtype0*
_output_shapes
: 

=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/MaximumMaximum<training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Prod_1?training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Maximum/y*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*
_output_shapes
: 

>training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/floordivFloorDiv:training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Prod=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Maximum*
_output_shapes
: *
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2
я
:training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/CastCast>training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/floordiv*

SrcT0*+
_class!
loc:@loss/dense_3_loss/Mean_2*
Truncate( *
_output_shapes
: *

DstT0

=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/truedivRealDiv:training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Tile:training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Cast*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*#
_output_shapes
:џџџџџџџџџ
Х
>training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/ShapeShapeloss/dense_3_loss/mul_1*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
out_type0*
_output_shapes
:*
T0
Г
@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Shape_1Const*
_output_shapes
: *.
_class$
" loc:@loss/dense_3_loss/truediv_1*
valueB *
dtype0
ж
Ntraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgs>training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Shape@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1

@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDivRealDiv=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/truedivloss/dense_3_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1
Х
<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/SumSum@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDivNtraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1
Е
@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/ReshapeReshape<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Sum>training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Shape*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
Tshape0*#
_output_shapes
:џџџџџџџџџ*
T0
К
<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/NegNegloss/dense_3_loss/mul_1*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ

Btraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDiv_1RealDiv<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Negloss/dense_3_loss/Mean_1*.
_class$
" loc:@loss/dense_3_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ*
T0

Btraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDiv_2RealDivBtraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDiv_1loss/dense_3_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1
Є
<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/mulMul=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/truedivBtraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDiv_2*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ
Х
>training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Sum_1Sum<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/mulPtraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
_output_shapes
:
Ў
Btraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Reshape_1Reshape>training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Sum_1@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Shape_1*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
Tshape0*
_output_shapes
: 
М
:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/ShapeShapeloss/dense_3_loss/Mean*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*
out_type0*
_output_shapes
:
О
<training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Shape_1Shapedense_3_sample_weights*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*
out_type0*
_output_shapes
:
Ц
Jtraining/Adam/gradients/loss/dense_3_loss/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Shape<training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Shape_1**
_class 
loc:@loss/dense_3_loss/mul_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ѓ
8training/Adam/gradients/loss/dense_3_loss/mul_1_grad/MulMul@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Reshapedense_3_sample_weights*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*#
_output_shapes
:џџџџџџџџџ
Б
8training/Adam/gradients/loss/dense_3_loss/mul_1_grad/SumSum8training/Adam/gradients/loss/dense_3_loss/mul_1_grad/MulJtraining/Adam/gradients/loss/dense_3_loss/mul_1_grad/BroadcastGradientArgs*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
Ѕ
<training/Adam/gradients/loss/dense_3_loss/mul_1_grad/ReshapeReshape8training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Sum:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Shape*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*
Tshape0*#
_output_shapes
:џџџџџџџџџ
ѕ
:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Mul_1Mulloss/dense_3_loss/Mean@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Reshape*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*#
_output_shapes
:џџџџџџџџџ
З
:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Sum_1Sum:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Mul_1Ltraining/Adam/gradients/loss/dense_3_loss/mul_1_grad/BroadcastGradientArgs:1*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
Ћ
>training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Reshape_1Reshape:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Sum_1<training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Shape_1*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*
Tshape0*#
_output_shapes
:џџџџџџџџџ
Й
9training/Adam/gradients/loss/dense_3_loss/Mean_grad/ShapeShapeloss/dense_3_loss/Neg*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
out_type0*
_output_shapes
:
Ѕ
8training/Adam/gradients/loss/dense_3_loss/Mean_grad/SizeConst*)
_class
loc:@loss/dense_3_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
№
7training/Adam/gradients/loss/dense_3_loss/Mean_grad/addAdd(loss/dense_3_loss/Mean/reduction_indices8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Size*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 

7training/Adam/gradients/loss/dense_3_loss/Mean_grad/modFloorMod7training/Adam/gradients/loss/dense_3_loss/Mean_grad/add8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Size*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
А
;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_1Const*)
_class
loc:@loss/dense_3_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
Ќ
?training/Adam/gradients/loss/dense_3_loss/Mean_grad/range/startConst*)
_class
loc:@loss/dense_3_loss/Mean*
value	B : *
dtype0*
_output_shapes
: 
Ќ
?training/Adam/gradients/loss/dense_3_loss/Mean_grad/range/deltaConst*)
_class
loc:@loss/dense_3_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
б
9training/Adam/gradients/loss/dense_3_loss/Mean_grad/rangeRange?training/Adam/gradients/loss/dense_3_loss/Mean_grad/range/start8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Size?training/Adam/gradients/loss/dense_3_loss/Mean_grad/range/delta*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:*

Tidx0
Ћ
>training/Adam/gradients/loss/dense_3_loss/Mean_grad/Fill/valueConst*
dtype0*
_output_shapes
: *)
_class
loc:@loss/dense_3_loss/Mean*
value	B :

8training/Adam/gradients/loss/dense_3_loss/Mean_grad/FillFill;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_1>training/Adam/gradients/loss/dense_3_loss/Mean_grad/Fill/value*
T0*)
_class
loc:@loss/dense_3_loss/Mean*

index_type0*
_output_shapes
: 

Atraining/Adam/gradients/loss/dense_3_loss/Mean_grad/DynamicStitchDynamicStitch9training/Adam/gradients/loss/dense_3_loss/Mean_grad/range7training/Adam/gradients/loss/dense_3_loss/Mean_grad/mod9training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Fill*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
N*
_output_shapes
:
Њ
=training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *)
_class
loc:@loss/dense_3_loss/Mean*
value	B :

;training/Adam/gradients/loss/dense_3_loss/Mean_grad/MaximumMaximumAtraining/Adam/gradients/loss/dense_3_loss/Mean_grad/DynamicStitch=training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum/y*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:

<training/Adam/gradients/loss/dense_3_loss/Mean_grad/floordivFloorDiv9training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
Ў
;training/Adam/gradients/loss/dense_3_loss/Mean_grad/ReshapeReshape<training/Adam/gradients/loss/dense_3_loss/mul_1_grad/ReshapeAtraining/Adam/gradients/loss/dense_3_loss/Mean_grad/DynamicStitch*#
_output_shapes
:џџџџџџџџџ*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
Tshape0
І
8training/Adam/gradients/loss/dense_3_loss/Mean_grad/TileTile;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Reshape<training/Adam/gradients/loss/dense_3_loss/Mean_grad/floordiv*

Tmultiples0*
T0*)
_class
loc:@loss/dense_3_loss/Mean*#
_output_shapes
:џџџџџџџџџ
Л
;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_2Shapeloss/dense_3_loss/Neg*
_output_shapes
:*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
out_type0
М
;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_3Shapeloss/dense_3_loss/Mean*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
out_type0*
_output_shapes
:
Ў
9training/Adam/gradients/loss/dense_3_loss/Mean_grad/ConstConst*)
_class
loc:@loss/dense_3_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
Ё
8training/Adam/gradients/loss/dense_3_loss/Mean_grad/ProdProd;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_29training/Adam/gradients/loss/dense_3_loss/Mean_grad/Const*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
А
;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Const_1Const*)
_class
loc:@loss/dense_3_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
Ѕ
:training/Adam/gradients/loss/dense_3_loss/Mean_grad/Prod_1Prod;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_3;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Ќ
?training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum_1/yConst*
_output_shapes
: *)
_class
loc:@loss/dense_3_loss/Mean*
value	B :*
dtype0

=training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum_1Maximum:training/Adam/gradients/loss/dense_3_loss/Mean_grad/Prod_1?training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum_1/y*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: *
T0

>training/Adam/gradients/loss/dense_3_loss/Mean_grad/floordiv_1FloorDiv8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Prod=training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum_1*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
ы
8training/Adam/gradients/loss/dense_3_loss/Mean_grad/CastCast>training/Adam/gradients/loss/dense_3_loss/Mean_grad/floordiv_1*)
_class
loc:@loss/dense_3_loss/Mean*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0

;training/Adam/gradients/loss/dense_3_loss/Mean_grad/truedivRealDiv8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Tile8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Cast*
T0*)
_class
loc:@loss/dense_3_loss/Mean*#
_output_shapes
:џџџџџџџџџ
в
6training/Adam/gradients/loss/dense_3_loss/Neg_grad/NegNeg;training/Adam/gradients/loss/dense_3_loss/Mean_grad/truediv*(
_class
loc:@loss/dense_3_loss/Neg*#
_output_shapes
:џџџџџџџџџ*
T0
Л
:training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/ShapeShapeloss/dense_3_loss/mul*
T0**
_class 
loc:@loss/dense_3_loss/Sum_1*
out_type0*
_output_shapes
:
Ї
9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/SizeConst**
_class 
loc:@loss/dense_3_loss/Sum_1*
value	B :*
dtype0*
_output_shapes
: 
ђ
8training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/addAdd)loss/dense_3_loss/Sum_1/reduction_indices9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Size*
_output_shapes
: *
T0**
_class 
loc:@loss/dense_3_loss/Sum_1

8training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/modFloorMod8training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/add9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Size*
T0**
_class 
loc:@loss/dense_3_loss/Sum_1*
_output_shapes
: 
Ћ
<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Shape_1Const**
_class 
loc:@loss/dense_3_loss/Sum_1*
valueB *
dtype0*
_output_shapes
: 
Ў
@training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/range/startConst**
_class 
loc:@loss/dense_3_loss/Sum_1*
value	B : *
dtype0*
_output_shapes
: 
Ў
@training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/range/deltaConst**
_class 
loc:@loss/dense_3_loss/Sum_1*
value	B :*
dtype0*
_output_shapes
: 
ж
:training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/rangeRange@training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/range/start9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Size@training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/range/delta*
_output_shapes
:*

Tidx0**
_class 
loc:@loss/dense_3_loss/Sum_1
­
?training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Fill/valueConst**
_class 
loc:@loss/dense_3_loss/Sum_1*
value	B :*
dtype0*
_output_shapes
: 

9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/FillFill<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Shape_1?training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Fill/value*
T0**
_class 
loc:@loss/dense_3_loss/Sum_1*

index_type0*
_output_shapes
: 

Btraining/Adam/gradients/loss/dense_3_loss/Sum_1_grad/DynamicStitchDynamicStitch:training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/range8training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/mod:training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Shape9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Fill*
T0**
_class 
loc:@loss/dense_3_loss/Sum_1*
N*
_output_shapes
:
Ќ
>training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Maximum/yConst**
_class 
loc:@loss/dense_3_loss/Sum_1*
value	B :*
dtype0*
_output_shapes
: 

<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/MaximumMaximumBtraining/Adam/gradients/loss/dense_3_loss/Sum_1_grad/DynamicStitch>training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Maximum/y*
T0**
_class 
loc:@loss/dense_3_loss/Sum_1*
_output_shapes
:

=training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/floordivFloorDiv:training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Shape<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Maximum*
_output_shapes
:*
T0**
_class 
loc:@loss/dense_3_loss/Sum_1
И
<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/ReshapeReshape6training/Adam/gradients/loss/dense_3_loss/Neg_grad/NegBtraining/Adam/gradients/loss/dense_3_loss/Sum_1_grad/DynamicStitch*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0**
_class 
loc:@loss/dense_3_loss/Sum_1*
Tshape0
Ў
9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/TileTile<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Reshape=training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/floordiv*
T0**
_class 
loc:@loss/dense_3_loss/Sum_1*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0
А
8training/Adam/gradients/loss/dense_3_loss/mul_grad/ShapeShapedense_3_target*
_output_shapes
:*
T0*(
_class
loc:@loss/dense_3_loss/mul*
out_type0
Й
:training/Adam/gradients/loss/dense_3_loss/mul_grad/Shape_1Shapeloss/dense_3_loss/Log*
T0*(
_class
loc:@loss/dense_3_loss/mul*
out_type0*
_output_shapes
:
О
Htraining/Adam/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8training/Adam/gradients/loss/dense_3_loss/mul_grad/Shape:training/Adam/gradients/loss/dense_3_loss/mul_grad/Shape_1*
T0*(
_class
loc:@loss/dense_3_loss/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ы
6training/Adam/gradients/loss/dense_3_loss/mul_grad/MulMul9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Tileloss/dense_3_loss/Log*
T0*(
_class
loc:@loss/dense_3_loss/mul*'
_output_shapes
:џџџџџџџџџ
Љ
6training/Adam/gradients/loss/dense_3_loss/mul_grad/SumSum6training/Adam/gradients/loss/dense_3_loss/mul_grad/MulHtraining/Adam/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgs*
T0*(
_class
loc:@loss/dense_3_loss/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
Њ
:training/Adam/gradients/loss/dense_3_loss/mul_grad/ReshapeReshape6training/Adam/gradients/loss/dense_3_loss/mul_grad/Sum8training/Adam/gradients/loss/dense_3_loss/mul_grad/Shape*
T0*(
_class
loc:@loss/dense_3_loss/mul*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
ц
8training/Adam/gradients/loss/dense_3_loss/mul_grad/Mul_1Muldense_3_target9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Tile*
T0*(
_class
loc:@loss/dense_3_loss/mul*'
_output_shapes
:џџџџџџџџџ
Џ
8training/Adam/gradients/loss/dense_3_loss/mul_grad/Sum_1Sum8training/Adam/gradients/loss/dense_3_loss/mul_grad/Mul_1Jtraining/Adam/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgs:1*(
_class
loc:@loss/dense_3_loss/mul*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ї
<training/Adam/gradients/loss/dense_3_loss/mul_grad/Reshape_1Reshape8training/Adam/gradients/loss/dense_3_loss/mul_grad/Sum_1:training/Adam/gradients/loss/dense_3_loss/mul_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*(
_class
loc:@loss/dense_3_loss/mul*
Tshape0

=training/Adam/gradients/loss/dense_3_loss/Log_grad/Reciprocal
Reciprocalloss/dense_3_loss/clip_by_value=^training/Adam/gradients/loss/dense_3_loss/mul_grad/Reshape_1*
T0*(
_class
loc:@loss/dense_3_loss/Log*'
_output_shapes
:џџџџџџџџџ

6training/Adam/gradients/loss/dense_3_loss/Log_grad/mulMul<training/Adam/gradients/loss/dense_3_loss/mul_grad/Reshape_1=training/Adam/gradients/loss/dense_3_loss/Log_grad/Reciprocal*'
_output_shapes
:џџџџџџџџџ*
T0*(
_class
loc:@loss/dense_3_loss/Log
н
Btraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/ShapeShape'loss/dense_3_loss/clip_by_value/Minimum*
_output_shapes
:*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
out_type0
Л
Dtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape_1Const*
_output_shapes
: *2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
valueB *
dtype0
ю
Dtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape_2Shape6training/Adam/gradients/loss/dense_3_loss/Log_grad/mul*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
out_type0*
_output_shapes
:
С
Htraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/zeros/ConstConst*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
valueB
 *    *
dtype0*
_output_shapes
: 
в
Btraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/zerosFillDtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape_2Htraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/zeros/Const*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*

index_type0*'
_output_shapes
:џџџџџџџџџ

Itraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/GreaterEqualGreaterEqual'loss/dense_3_loss/clip_by_value/Minimumloss/dense_3_loss/Const*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*'
_output_shapes
:џџџџџџџџџ
ц
Rtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsBtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/ShapeDtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape_1*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
њ
Ctraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/SelectSelectItraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/GreaterEqual6training/Adam/gradients/loss/dense_3_loss/Log_grad/mulBtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/zeros*'
_output_shapes
:џџџџџџџџџ*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value
ќ
Etraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Select_1SelectItraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/GreaterEqualBtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/zeros6training/Adam/gradients/loss/dense_3_loss/Log_grad/mul*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*'
_output_shapes
:џџџџџџџџџ
д
@training/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/SumSumCtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/SelectRtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/BroadcastGradientArgs*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
_output_shapes
:*
	keep_dims( *

Tidx0
Щ
Dtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/ReshapeReshape@training/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/SumBtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
Tshape0
к
Btraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Sum_1SumEtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Select_1Ttraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/BroadcastGradientArgs:1*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
_output_shapes
:*
	keep_dims( *

Tidx0
О
Ftraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Reshape_1ReshapeBtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Sum_1Dtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape_1*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
Tshape0*
_output_shapes
: 
п
Jtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/ShapeShapeloss/dense_3_loss/truediv*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
out_type0*
_output_shapes
:
Ы
Ltraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape_1Const*
_output_shapes
: *:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
valueB *
dtype0

Ltraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape_2ShapeDtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Reshape*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
out_type0*
_output_shapes
:
б
Ptraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/zeros/ConstConst*
_output_shapes
: *:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
valueB
 *    *
dtype0
ђ
Jtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/zerosFillLtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape_2Ptraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/zeros/Const*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*

index_type0*'
_output_shapes
:џџџџџџџџџ
ћ
Ntraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/LessEqual	LessEqualloss/dense_3_loss/truedivloss/dense_3_loss/sub*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ

Ztraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/ShapeLtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape_1*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ѕ
Ktraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/SelectSelectNtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/LessEqualDtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/ReshapeJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/zeros*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ
Ї
Mtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Select_1SelectNtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/LessEqualJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/zerosDtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum
є
Htraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/SumSumKtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/SelectZtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum
щ
Ltraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/ReshapeReshapeHtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/SumJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
Tshape0*'
_output_shapes
:џџџџџџџџџ
њ
Jtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Sum_1SumMtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Select_1\training/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
о
Ntraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Reshape_1ReshapeJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Sum_1Ltraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape_1*
_output_shapes
: *
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
Tshape0
Й
<training/Adam/gradients/loss/dense_3_loss/truediv_grad/ShapeShapedense_3/Softmax*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*
out_type0*
_output_shapes
:
С
>training/Adam/gradients/loss/dense_3_loss/truediv_grad/Shape_1Shapeloss/dense_3_loss/Sum*,
_class"
 loc:@loss/dense_3_loss/truediv*
out_type0*
_output_shapes
:*
T0
Ю
Ltraining/Adam/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs<training/Adam/gradients/loss/dense_3_loss/truediv_grad/Shape>training/Adam/gradients/loss/dense_3_loss/truediv_grad/Shape_1*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

>training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDivRealDivLtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Reshapeloss/dense_3_loss/Sum*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*'
_output_shapes
:џџџџџџџџџ
Н
:training/Adam/gradients/loss/dense_3_loss/truediv_grad/SumSum>training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDivLtraining/Adam/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
:
Б
>training/Adam/gradients/loss/dense_3_loss/truediv_grad/ReshapeReshape:training/Adam/gradients/loss/dense_3_loss/truediv_grad/Sum<training/Adam/gradients/loss/dense_3_loss/truediv_grad/Shape*,
_class"
 loc:@loss/dense_3_loss/truediv*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
В
:training/Adam/gradients/loss/dense_3_loss/truediv_grad/NegNegdense_3/Softmax*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*'
_output_shapes
:џџџџџџџџџ
ў
@training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDiv_1RealDiv:training/Adam/gradients/loss/dense_3_loss/truediv_grad/Negloss/dense_3_loss/Sum*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*'
_output_shapes
:џџџџџџџџџ

@training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDiv_2RealDiv@training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDiv_1loss/dense_3_loss/Sum*'
_output_shapes
:џџџџџџџџџ*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv
Б
:training/Adam/gradients/loss/dense_3_loss/truediv_grad/mulMulLtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Reshape@training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDiv_2*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*'
_output_shapes
:џџџџџџџџџ
Н
<training/Adam/gradients/loss/dense_3_loss/truediv_grad/Sum_1Sum:training/Adam/gradients/loss/dense_3_loss/truediv_grad/mulNtraining/Adam/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgs:1*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
:*
	keep_dims( *

Tidx0
З
@training/Adam/gradients/loss/dense_3_loss/truediv_grad/Reshape_1Reshape<training/Adam/gradients/loss/dense_3_loss/truediv_grad/Sum_1>training/Adam/gradients/loss/dense_3_loss/truediv_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*
Tshape0
Б
8training/Adam/gradients/loss/dense_3_loss/Sum_grad/ShapeShapedense_3/Softmax*
T0*(
_class
loc:@loss/dense_3_loss/Sum*
out_type0*
_output_shapes
:
Ѓ
7training/Adam/gradients/loss/dense_3_loss/Sum_grad/SizeConst*(
_class
loc:@loss/dense_3_loss/Sum*
value	B :*
dtype0*
_output_shapes
: 
ъ
6training/Adam/gradients/loss/dense_3_loss/Sum_grad/addAdd'loss/dense_3_loss/Sum/reduction_indices7training/Adam/gradients/loss/dense_3_loss/Sum_grad/Size*
_output_shapes
: *
T0*(
_class
loc:@loss/dense_3_loss/Sum
ў
6training/Adam/gradients/loss/dense_3_loss/Sum_grad/modFloorMod6training/Adam/gradients/loss/dense_3_loss/Sum_grad/add7training/Adam/gradients/loss/dense_3_loss/Sum_grad/Size*
T0*(
_class
loc:@loss/dense_3_loss/Sum*
_output_shapes
: 
Ї
:training/Adam/gradients/loss/dense_3_loss/Sum_grad/Shape_1Const*(
_class
loc:@loss/dense_3_loss/Sum*
valueB *
dtype0*
_output_shapes
: 
Њ
>training/Adam/gradients/loss/dense_3_loss/Sum_grad/range/startConst*
dtype0*
_output_shapes
: *(
_class
loc:@loss/dense_3_loss/Sum*
value	B : 
Њ
>training/Adam/gradients/loss/dense_3_loss/Sum_grad/range/deltaConst*
dtype0*
_output_shapes
: *(
_class
loc:@loss/dense_3_loss/Sum*
value	B :
Ь
8training/Adam/gradients/loss/dense_3_loss/Sum_grad/rangeRange>training/Adam/gradients/loss/dense_3_loss/Sum_grad/range/start7training/Adam/gradients/loss/dense_3_loss/Sum_grad/Size>training/Adam/gradients/loss/dense_3_loss/Sum_grad/range/delta*

Tidx0*(
_class
loc:@loss/dense_3_loss/Sum*
_output_shapes
:
Љ
=training/Adam/gradients/loss/dense_3_loss/Sum_grad/Fill/valueConst*(
_class
loc:@loss/dense_3_loss/Sum*
value	B :*
dtype0*
_output_shapes
: 

7training/Adam/gradients/loss/dense_3_loss/Sum_grad/FillFill:training/Adam/gradients/loss/dense_3_loss/Sum_grad/Shape_1=training/Adam/gradients/loss/dense_3_loss/Sum_grad/Fill/value*(
_class
loc:@loss/dense_3_loss/Sum*

index_type0*
_output_shapes
: *
T0

@training/Adam/gradients/loss/dense_3_loss/Sum_grad/DynamicStitchDynamicStitch8training/Adam/gradients/loss/dense_3_loss/Sum_grad/range6training/Adam/gradients/loss/dense_3_loss/Sum_grad/mod8training/Adam/gradients/loss/dense_3_loss/Sum_grad/Shape7training/Adam/gradients/loss/dense_3_loss/Sum_grad/Fill*
T0*(
_class
loc:@loss/dense_3_loss/Sum*
N*
_output_shapes
:
Ј
<training/Adam/gradients/loss/dense_3_loss/Sum_grad/Maximum/yConst*(
_class
loc:@loss/dense_3_loss/Sum*
value	B :*
dtype0*
_output_shapes
: 

:training/Adam/gradients/loss/dense_3_loss/Sum_grad/MaximumMaximum@training/Adam/gradients/loss/dense_3_loss/Sum_grad/DynamicStitch<training/Adam/gradients/loss/dense_3_loss/Sum_grad/Maximum/y*
T0*(
_class
loc:@loss/dense_3_loss/Sum*
_output_shapes
:

;training/Adam/gradients/loss/dense_3_loss/Sum_grad/floordivFloorDiv8training/Adam/gradients/loss/dense_3_loss/Sum_grad/Shape:training/Adam/gradients/loss/dense_3_loss/Sum_grad/Maximum*
T0*(
_class
loc:@loss/dense_3_loss/Sum*
_output_shapes
:
М
:training/Adam/gradients/loss/dense_3_loss/Sum_grad/ReshapeReshape@training/Adam/gradients/loss/dense_3_loss/truediv_grad/Reshape_1@training/Adam/gradients/loss/dense_3_loss/Sum_grad/DynamicStitch*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0*(
_class
loc:@loss/dense_3_loss/Sum*
Tshape0
І
7training/Adam/gradients/loss/dense_3_loss/Sum_grad/TileTile:training/Adam/gradients/loss/dense_3_loss/Sum_grad/Reshape;training/Adam/gradients/loss/dense_3_loss/Sum_grad/floordiv*
T0*(
_class
loc:@loss/dense_3_loss/Sum*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0

training/Adam/gradients/AddNAddN>training/Adam/gradients/loss/dense_3_loss/truediv_grad/Reshape7training/Adam/gradients/loss/dense_3_loss/Sum_grad/Tile*
N*'
_output_shapes
:џџџџџџџџџ*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv
М
0training/Adam/gradients/dense_3/Softmax_grad/mulMultraining/Adam/gradients/AddNdense_3/Softmax*
T0*"
_class
loc:@dense_3/Softmax*'
_output_shapes
:џџџџџџџџџ
Б
Btraining/Adam/gradients/dense_3/Softmax_grad/Sum/reduction_indicesConst*"
_class
loc:@dense_3/Softmax*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
 
0training/Adam/gradients/dense_3/Softmax_grad/SumSum0training/Adam/gradients/dense_3/Softmax_grad/mulBtraining/Adam/gradients/dense_3/Softmax_grad/Sum/reduction_indices*
T0*"
_class
loc:@dense_3/Softmax*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0
н
0training/Adam/gradients/dense_3/Softmax_grad/subSubtraining/Adam/gradients/AddN0training/Adam/gradients/dense_3/Softmax_grad/Sum*
T0*"
_class
loc:@dense_3/Softmax*'
_output_shapes
:џџџџџџџџџ
в
2training/Adam/gradients/dense_3/Softmax_grad/mul_1Mul0training/Adam/gradients/dense_3/Softmax_grad/subdense_3/Softmax*"
_class
loc:@dense_3/Softmax*'
_output_shapes
:џџџџџџџџџ*
T0
л
8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_3/Softmax_grad/mul_1*
T0*"
_class
loc:@dense_3/BiasAdd*
data_formatNHWC*
_output_shapes
:

2training/Adam/gradients/dense_3/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_3/Softmax_grad/mul_1dense_3/kernel/read*
T0*!
_class
loc:@dense_3/MatMul*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
ћ
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldropout_3/cond/Merge2training/Adam/gradients/dense_3/Softmax_grad/mul_1*
T0*!
_class
loc:@dense_3/MatMul*
_output_shapes
:	*
transpose_a(*
transpose_b( 
ћ
;training/Adam/gradients/dropout_3/cond/Merge_grad/cond_gradSwitch2training/Adam/gradients/dense_3/MatMul_grad/MatMuldropout_3/cond/pred_id*
T0*!
_class
loc:@dense_3/MatMul*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
И
 training/Adam/gradients/Switch_2Switchdense_2/Reludropout_3/cond/pred_id*
T0*
_class
loc:@dense_2/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
Є
 training/Adam/gradients/IdentityIdentity"training/Adam/gradients/Switch_2:1*(
_output_shapes
:џџџџџџџџџ*
T0*
_class
loc:@dense_2/Relu
Ђ
training/Adam/gradients/Shape_1Shape"training/Adam/gradients/Switch_2:1*
T0*
_class
loc:@dense_2/Relu*
out_type0*
_output_shapes
:
Ќ
#training/Adam/gradients/zeros/ConstConst!^training/Adam/gradients/Identity*
dtype0*
_output_shapes
: *
_class
loc:@dense_2/Relu*
valueB
 *    
б
training/Adam/gradients/zerosFilltraining/Adam/gradients/Shape_1#training/Adam/gradients/zeros/Const*
T0*
_class
loc:@dense_2/Relu*

index_type0*(
_output_shapes
:џџџџџџџџџ

>training/Adam/gradients/dropout_3/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_3/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros*
N**
_output_shapes
:џџџџџџџџџ: *
T0*
_class
loc:@dense_2/Relu
Ц
=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/ShapeShapedropout_3/cond/dropout/div*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*
out_type0*
_output_shapes
:
Ъ
?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Shape_1Shapedropout_3/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*
out_type0*
_output_shapes
:
в
Mtraining/Adam/gradients/dropout_3/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Shape?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

;training/Adam/gradients/dropout_3/cond/dropout/mul_grad/MulMul=training/Adam/gradients/dropout_3/cond/Merge_grad/cond_grad:1dropout_3/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*(
_output_shapes
:џџџџџџџџџ
Н
;training/Adam/gradients/dropout_3/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_3/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_3/cond/dropout/mul_grad/BroadcastGradientArgs*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*
Tshape0

=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Mul_1Muldropout_3/cond/dropout/div=training/Adam/gradients/dropout_3/cond/Merge_grad/cond_grad:1*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*(
_output_shapes
:џџџџџџџџџ
У
=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_3/cond/dropout/mul_grad/BroadcastGradientArgs:1*-
_class#
!loc:@dropout_3/cond/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
М
Atraining/Adam/gradients/dropout_3/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*
Tshape0*(
_output_shapes
:џџџџџџџџџ
О
=training/Adam/gradients/dropout_3/cond/dropout/div_grad/ShapeShapedropout_3/cond/mul*
T0*-
_class#
!loc:@dropout_3/cond/dropout/div*
out_type0*
_output_shapes
:
Б
?training/Adam/gradients/dropout_3/cond/dropout/div_grad/Shape_1Const*-
_class#
!loc:@dropout_3/cond/dropout/div*
valueB *
dtype0*
_output_shapes
: 
в
Mtraining/Adam/gradients/dropout_3/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_3/cond/dropout/div_grad/Shape?training/Adam/gradients/dropout_3/cond/dropout/div_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_3/cond/dropout/div

?training/Adam/gradients/dropout_3/cond/dropout/div_grad/RealDivRealDiv?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Reshape dropout_3/cond/dropout/keep_prob*-
_class#
!loc:@dropout_3/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ*
T0
С
;training/Adam/gradients/dropout_3/cond/dropout/div_grad/SumSum?training/Adam/gradients/dropout_3/cond/dropout/div_grad/RealDivMtraining/Adam/gradients/dropout_3/cond/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_3/cond/dropout/div
Ж
?training/Adam/gradients/dropout_3/cond/dropout/div_grad/ReshapeReshape;training/Adam/gradients/dropout_3/cond/dropout/div_grad/Sum=training/Adam/gradients/dropout_3/cond/dropout/div_grad/Shape*-
_class#
!loc:@dropout_3/cond/dropout/div*
Tshape0*(
_output_shapes
:џџџџџџџџџ*
T0
И
;training/Adam/gradients/dropout_3/cond/dropout/div_grad/NegNegdropout_3/cond/mul*
T0*-
_class#
!loc:@dropout_3/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ

Atraining/Adam/gradients/dropout_3/cond/dropout/div_grad/RealDiv_1RealDiv;training/Adam/gradients/dropout_3/cond/dropout/div_grad/Neg dropout_3/cond/dropout/keep_prob*
T0*-
_class#
!loc:@dropout_3/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ

Atraining/Adam/gradients/dropout_3/cond/dropout/div_grad/RealDiv_2RealDivAtraining/Adam/gradients/dropout_3/cond/dropout/div_grad/RealDiv_1 dropout_3/cond/dropout/keep_prob*
T0*-
_class#
!loc:@dropout_3/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ
Ј
;training/Adam/gradients/dropout_3/cond/dropout/div_grad/mulMul?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/ReshapeAtraining/Adam/gradients/dropout_3/cond/dropout/div_grad/RealDiv_2*(
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_3/cond/dropout/div
С
=training/Adam/gradients/dropout_3/cond/dropout/div_grad/Sum_1Sum;training/Adam/gradients/dropout_3/cond/dropout/div_grad/mulOtraining/Adam/gradients/dropout_3/cond/dropout/div_grad/BroadcastGradientArgs:1*
T0*-
_class#
!loc:@dropout_3/cond/dropout/div*
_output_shapes
:*
	keep_dims( *

Tidx0
Њ
Atraining/Adam/gradients/dropout_3/cond/dropout/div_grad/Reshape_1Reshape=training/Adam/gradients/dropout_3/cond/dropout/div_grad/Sum_1?training/Adam/gradients/dropout_3/cond/dropout/div_grad/Shape_1*
T0*-
_class#
!loc:@dropout_3/cond/dropout/div*
Tshape0*
_output_shapes
: 
З
5training/Adam/gradients/dropout_3/cond/mul_grad/ShapeShapedropout_3/cond/mul/Switch:1*
T0*%
_class
loc:@dropout_3/cond/mul*
out_type0*
_output_shapes
:
Ё
7training/Adam/gradients/dropout_3/cond/mul_grad/Shape_1Const*%
_class
loc:@dropout_3/cond/mul*
valueB *
dtype0*
_output_shapes
: 
В
Etraining/Adam/gradients/dropout_3/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/dropout_3/cond/mul_grad/Shape7training/Adam/gradients/dropout_3/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_3/cond/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ы
3training/Adam/gradients/dropout_3/cond/mul_grad/MulMul?training/Adam/gradients/dropout_3/cond/dropout/div_grad/Reshapedropout_3/cond/mul/y*
T0*%
_class
loc:@dropout_3/cond/mul*(
_output_shapes
:џџџџџџџџџ

3training/Adam/gradients/dropout_3/cond/mul_grad/SumSum3training/Adam/gradients/dropout_3/cond/mul_grad/MulEtraining/Adam/gradients/dropout_3/cond/mul_grad/BroadcastGradientArgs*
T0*%
_class
loc:@dropout_3/cond/mul*
_output_shapes
:*
	keep_dims( *

Tidx0

7training/Adam/gradients/dropout_3/cond/mul_grad/ReshapeReshape3training/Adam/gradients/dropout_3/cond/mul_grad/Sum5training/Adam/gradients/dropout_3/cond/mul_grad/Shape*
T0*%
_class
loc:@dropout_3/cond/mul*
Tshape0*(
_output_shapes
:џџџџџџџџџ
є
5training/Adam/gradients/dropout_3/cond/mul_grad/Mul_1Muldropout_3/cond/mul/Switch:1?training/Adam/gradients/dropout_3/cond/dropout/div_grad/Reshape*(
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@dropout_3/cond/mul
Ѓ
5training/Adam/gradients/dropout_3/cond/mul_grad/Sum_1Sum5training/Adam/gradients/dropout_3/cond/mul_grad/Mul_1Gtraining/Adam/gradients/dropout_3/cond/mul_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@dropout_3/cond/mul*
_output_shapes
:*
	keep_dims( *

Tidx0

9training/Adam/gradients/dropout_3/cond/mul_grad/Reshape_1Reshape5training/Adam/gradients/dropout_3/cond/mul_grad/Sum_17training/Adam/gradients/dropout_3/cond/mul_grad/Shape_1*%
_class
loc:@dropout_3/cond/mul*
Tshape0*
_output_shapes
: *
T0
И
 training/Adam/gradients/Switch_3Switchdense_2/Reludropout_3/cond/pred_id*
T0*
_class
loc:@dense_2/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
Є
"training/Adam/gradients/Identity_1Identity training/Adam/gradients/Switch_3*(
_output_shapes
:џџџџџџџџџ*
T0*
_class
loc:@dense_2/Relu
 
training/Adam/gradients/Shape_2Shape training/Adam/gradients/Switch_3*
T0*
_class
loc:@dense_2/Relu*
out_type0*
_output_shapes
:
А
%training/Adam/gradients/zeros_1/ConstConst#^training/Adam/gradients/Identity_1*
_class
loc:@dense_2/Relu*
valueB
 *    *
dtype0*
_output_shapes
: 
е
training/Adam/gradients/zeros_1Filltraining/Adam/gradients/Shape_2%training/Adam/gradients/zeros_1/Const*
T0*
_class
loc:@dense_2/Relu*

index_type0*(
_output_shapes
:џџџџџџџџџ

@training/Adam/gradients/dropout_3/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_17training/Adam/gradients/dropout_3/cond/mul_grad/Reshape*
_class
loc:@dense_2/Relu*
N**
_output_shapes
:џџџџџџџџџ: *
T0

training/Adam/gradients/AddN_1AddN>training/Adam/gradients/dropout_3/cond/Switch_1_grad/cond_grad@training/Adam/gradients/dropout_3/cond/mul/Switch_grad/cond_grad*
T0*
_class
loc:@dense_2/Relu*
N*(
_output_shapes
:џџџџџџџџџ
Р
2training/Adam/gradients/dense_2/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_1dense_2/Relu*
T0*
_class
loc:@dense_2/Relu*(
_output_shapes
:џџџџџџџџџ
м
8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
T0*"
_class
loc:@dense_2/BiasAdd*
data_formatNHWC*
_output_shapes	
:

2training/Adam/gradients/dense_2/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_2/Relu_grad/ReluGraddense_2/kernel/read*
T0*!
_class
loc:@dense_2/MatMul*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
ќ
4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1MatMuldropout_2/cond/Merge2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
T0*!
_class
loc:@dense_2/MatMul* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
ћ
;training/Adam/gradients/dropout_2/cond/Merge_grad/cond_gradSwitch2training/Adam/gradients/dense_2/MatMul_grad/MatMuldropout_2/cond/pred_id*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*
T0*!
_class
loc:@dense_2/MatMul
И
 training/Adam/gradients/Switch_4Switchdense_1/Reludropout_2/cond/pred_id*
T0*
_class
loc:@dense_1/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
І
"training/Adam/gradients/Identity_2Identity"training/Adam/gradients/Switch_4:1*
T0*
_class
loc:@dense_1/Relu*(
_output_shapes
:џџџџџџџџџ
Ђ
training/Adam/gradients/Shape_3Shape"training/Adam/gradients/Switch_4:1*
T0*
_class
loc:@dense_1/Relu*
out_type0*
_output_shapes
:
А
%training/Adam/gradients/zeros_2/ConstConst#^training/Adam/gradients/Identity_2*
dtype0*
_output_shapes
: *
_class
loc:@dense_1/Relu*
valueB
 *    
е
training/Adam/gradients/zeros_2Filltraining/Adam/gradients/Shape_3%training/Adam/gradients/zeros_2/Const*
T0*
_class
loc:@dense_1/Relu*

index_type0*(
_output_shapes
:џџџџџџџџџ

>training/Adam/gradients/dropout_2/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_2/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros_2**
_output_shapes
:џџџџџџџџџ: *
T0*
_class
loc:@dense_1/Relu*
N
Ц
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ShapeShapedropout_2/cond/dropout/div*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
out_type0*
_output_shapes
:
Ъ
?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1Shapedropout_2/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
out_type0*
_output_shapes
:
в
Mtraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*-
_class#
!loc:@dropout_2/cond/dropout/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/MulMul=training/Adam/gradients/dropout_2/cond/Merge_grad/cond_grad:1dropout_2/cond/dropout/Floor*(
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul
Н
;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul
Ж
?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
Tshape0*(
_output_shapes
:џџџџџџџџџ

=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Muldropout_2/cond/dropout/div=training/Adam/gradients/dropout_2/cond/Merge_grad/cond_grad:1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*(
_output_shapes
:џџџџџџџџџ
У
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul
М
Atraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*-
_class#
!loc:@dropout_2/cond/dropout/mul*
Tshape0*(
_output_shapes
:џџџџџџџџџ*
T0
О
=training/Adam/gradients/dropout_2/cond/dropout/div_grad/ShapeShapedropout_2/cond/mul*
_output_shapes
:*
T0*-
_class#
!loc:@dropout_2/cond/dropout/div*
out_type0
Б
?training/Adam/gradients/dropout_2/cond/dropout/div_grad/Shape_1Const*-
_class#
!loc:@dropout_2/cond/dropout/div*
valueB *
dtype0*
_output_shapes
: 
в
Mtraining/Adam/gradients/dropout_2/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_2/cond/dropout/div_grad/Shape?training/Adam/gradients/dropout_2/cond/dropout/div_grad/Shape_1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/div*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

?training/Adam/gradients/dropout_2/cond/dropout/div_grad/RealDivRealDiv?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Reshape dropout_2/cond/dropout/keep_prob*
T0*-
_class#
!loc:@dropout_2/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ
С
;training/Adam/gradients/dropout_2/cond/dropout/div_grad/SumSum?training/Adam/gradients/dropout_2/cond/dropout/div_grad/RealDivMtraining/Adam/gradients/dropout_2/cond/dropout/div_grad/BroadcastGradientArgs*
T0*-
_class#
!loc:@dropout_2/cond/dropout/div*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
?training/Adam/gradients/dropout_2/cond/dropout/div_grad/ReshapeReshape;training/Adam/gradients/dropout_2/cond/dropout/div_grad/Sum=training/Adam/gradients/dropout_2/cond/dropout/div_grad/Shape*
T0*-
_class#
!loc:@dropout_2/cond/dropout/div*
Tshape0*(
_output_shapes
:џџџџџџџџџ
И
;training/Adam/gradients/dropout_2/cond/dropout/div_grad/NegNegdropout_2/cond/mul*-
_class#
!loc:@dropout_2/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ*
T0

Atraining/Adam/gradients/dropout_2/cond/dropout/div_grad/RealDiv_1RealDiv;training/Adam/gradients/dropout_2/cond/dropout/div_grad/Neg dropout_2/cond/dropout/keep_prob*
T0*-
_class#
!loc:@dropout_2/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ

Atraining/Adam/gradients/dropout_2/cond/dropout/div_grad/RealDiv_2RealDivAtraining/Adam/gradients/dropout_2/cond/dropout/div_grad/RealDiv_1 dropout_2/cond/dropout/keep_prob*
T0*-
_class#
!loc:@dropout_2/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ
Ј
;training/Adam/gradients/dropout_2/cond/dropout/div_grad/mulMul?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ReshapeAtraining/Adam/gradients/dropout_2/cond/dropout/div_grad/RealDiv_2*
T0*-
_class#
!loc:@dropout_2/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ
С
=training/Adam/gradients/dropout_2/cond/dropout/div_grad/Sum_1Sum;training/Adam/gradients/dropout_2/cond/dropout/div_grad/mulOtraining/Adam/gradients/dropout_2/cond/dropout/div_grad/BroadcastGradientArgs:1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/div*
_output_shapes
:*
	keep_dims( *

Tidx0
Њ
Atraining/Adam/gradients/dropout_2/cond/dropout/div_grad/Reshape_1Reshape=training/Adam/gradients/dropout_2/cond/dropout/div_grad/Sum_1?training/Adam/gradients/dropout_2/cond/dropout/div_grad/Shape_1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/div*
Tshape0*
_output_shapes
: 
З
5training/Adam/gradients/dropout_2/cond/mul_grad/ShapeShapedropout_2/cond/mul/Switch:1*
T0*%
_class
loc:@dropout_2/cond/mul*
out_type0*
_output_shapes
:
Ё
7training/Adam/gradients/dropout_2/cond/mul_grad/Shape_1Const*
_output_shapes
: *%
_class
loc:@dropout_2/cond/mul*
valueB *
dtype0
В
Etraining/Adam/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/dropout_2/cond/mul_grad/Shape7training/Adam/gradients/dropout_2/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_2/cond/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ы
3training/Adam/gradients/dropout_2/cond/mul_grad/MulMul?training/Adam/gradients/dropout_2/cond/dropout/div_grad/Reshapedropout_2/cond/mul/y*%
_class
loc:@dropout_2/cond/mul*(
_output_shapes
:џџџџџџџџџ*
T0

3training/Adam/gradients/dropout_2/cond/mul_grad/SumSum3training/Adam/gradients/dropout_2/cond/mul_grad/MulEtraining/Adam/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgs*
T0*%
_class
loc:@dropout_2/cond/mul*
_output_shapes
:*
	keep_dims( *

Tidx0

7training/Adam/gradients/dropout_2/cond/mul_grad/ReshapeReshape3training/Adam/gradients/dropout_2/cond/mul_grad/Sum5training/Adam/gradients/dropout_2/cond/mul_grad/Shape*
T0*%
_class
loc:@dropout_2/cond/mul*
Tshape0*(
_output_shapes
:џџџџџџџџџ
є
5training/Adam/gradients/dropout_2/cond/mul_grad/Mul_1Muldropout_2/cond/mul/Switch:1?training/Adam/gradients/dropout_2/cond/dropout/div_grad/Reshape*(
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@dropout_2/cond/mul
Ѓ
5training/Adam/gradients/dropout_2/cond/mul_grad/Sum_1Sum5training/Adam/gradients/dropout_2/cond/mul_grad/Mul_1Gtraining/Adam/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgs:1*%
_class
loc:@dropout_2/cond/mul*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

9training/Adam/gradients/dropout_2/cond/mul_grad/Reshape_1Reshape5training/Adam/gradients/dropout_2/cond/mul_grad/Sum_17training/Adam/gradients/dropout_2/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_2/cond/mul*
Tshape0*
_output_shapes
: 
И
 training/Adam/gradients/Switch_5Switchdense_1/Reludropout_2/cond/pred_id*
_class
loc:@dense_1/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*
T0
Є
"training/Adam/gradients/Identity_3Identity training/Adam/gradients/Switch_5*
T0*
_class
loc:@dense_1/Relu*(
_output_shapes
:џџџџџџџџџ
 
training/Adam/gradients/Shape_4Shape training/Adam/gradients/Switch_5*
T0*
_class
loc:@dense_1/Relu*
out_type0*
_output_shapes
:
А
%training/Adam/gradients/zeros_3/ConstConst#^training/Adam/gradients/Identity_3*
_class
loc:@dense_1/Relu*
valueB
 *    *
dtype0*
_output_shapes
: 
е
training/Adam/gradients/zeros_3Filltraining/Adam/gradients/Shape_4%training/Adam/gradients/zeros_3/Const*
T0*
_class
loc:@dense_1/Relu*

index_type0*(
_output_shapes
:џџџџџџџџџ

@training/Adam/gradients/dropout_2/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_37training/Adam/gradients/dropout_2/cond/mul_grad/Reshape**
_output_shapes
:џџџџџџџџџ: *
T0*
_class
loc:@dense_1/Relu*
N

training/Adam/gradients/AddN_2AddN>training/Adam/gradients/dropout_2/cond/Switch_1_grad/cond_grad@training/Adam/gradients/dropout_2/cond/mul/Switch_grad/cond_grad*
_class
loc:@dense_1/Relu*
N*(
_output_shapes
:џџџџџџџџџ*
T0
Р
2training/Adam/gradients/dense_1/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_2dense_1/Relu*
T0*
_class
loc:@dense_1/Relu*(
_output_shapes
:џџџџџџџџџ
м
8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_1/Relu_grad/ReluGrad*
_output_shapes	
:*
T0*"
_class
loc:@dense_1/BiasAdd*
data_formatNHWC

2training/Adam/gradients/dense_1/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_1/Relu_grad/ReluGraddense_1/kernel/read*
T0*!
_class
loc:@dense_1/MatMul*(
_output_shapes
:џџџџџџџџџЪ*
transpose_a( *
transpose_b(
љ
4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1MatMulflatten_1/Reshape2training/Adam/gradients/dense_1/Relu_grad/ReluGrad*
T0*!
_class
loc:@dense_1/MatMul* 
_output_shapes
:
Ъ*
transpose_a(*
transpose_b( 
Ў
4training/Adam/gradients/flatten_1/Reshape_grad/ShapeShapedropout_1/cond/Merge*
_output_shapes
:*
T0*$
_class
loc:@flatten_1/Reshape*
out_type0

6training/Adam/gradients/flatten_1/Reshape_grad/ReshapeReshape2training/Adam/gradients/dense_1/MatMul_grad/MatMul4training/Adam/gradients/flatten_1/Reshape_grad/Shape*
T0*$
_class
loc:@flatten_1/Reshape*
Tshape0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

;training/Adam/gradients/dropout_1/cond/Merge_grad/cond_gradSwitch6training/Adam/gradients/flatten_1/Reshape_grad/Reshapedropout_1/cond/pred_id*
T0*$
_class
loc:@flatten_1/Reshape*T
_output_shapesB
@:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ
м
 training/Adam/gradients/Switch_6Switchlstm_1/transpose_1dropout_1/cond/pred_id*
T0*%
_class
loc:@lstm_1/transpose_1*T
_output_shapesB
@:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ
И
"training/Adam/gradients/Identity_4Identity"training/Adam/gradients/Switch_6:1*%
_class
loc:@lstm_1/transpose_1*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0
Ј
training/Adam/gradients/Shape_5Shape"training/Adam/gradients/Switch_6:1*
T0*%
_class
loc:@lstm_1/transpose_1*
out_type0*
_output_shapes
:
Ж
%training/Adam/gradients/zeros_4/ConstConst#^training/Adam/gradients/Identity_4*%
_class
loc:@lstm_1/transpose_1*
valueB
 *    *
dtype0*
_output_shapes
: 
ч
training/Adam/gradients/zeros_4Filltraining/Adam/gradients/Shape_5%training/Adam/gradients/zeros_4/Const*
T0*%
_class
loc:@lstm_1/transpose_1*

index_type0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

>training/Adam/gradients/dropout_1/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_1/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros_4*
T0*%
_class
loc:@lstm_1/transpose_1*
N*6
_output_shapes$
":џџџџџџџџџџџџџџџџџџ: 
Ц
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ShapeShapedropout_1/cond/dropout/div*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
out_type0*
_output_shapes
:
Ъ
?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1Shapedropout_1/cond/dropout/Floor*
_output_shapes
:*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
out_type0
в
Mtraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/MulMul=training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad:1dropout_1/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
Н
;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
Т
?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape*-
_class#
!loc:@dropout_1/cond/dropout/mul*
Tshape0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0

=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Muldropout_1/cond/dropout/div=training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad:1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
У
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:
Ш
Atraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
Tshape0
О
=training/Adam/gradients/dropout_1/cond/dropout/div_grad/ShapeShapedropout_1/cond/mul*
_output_shapes
:*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*
out_type0
Б
?training/Adam/gradients/dropout_1/cond/dropout/div_grad/Shape_1Const*-
_class#
!loc:@dropout_1/cond/dropout/div*
valueB *
dtype0*
_output_shapes
: 
в
Mtraining/Adam/gradients/dropout_1/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_1/cond/dropout/div_grad/Shape?training/Adam/gradients/dropout_1/cond/dropout/div_grad/Shape_1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

?training/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDivRealDiv?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Reshape dropout_1/cond/dropout/keep_prob*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div
С
;training/Adam/gradients/dropout_1/cond/dropout/div_grad/SumSum?training/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDivMtraining/Adam/gradients/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*
_output_shapes
:
Т
?training/Adam/gradients/dropout_1/cond/dropout/div_grad/ReshapeReshape;training/Adam/gradients/dropout_1/cond/dropout/div_grad/Sum=training/Adam/gradients/dropout_1/cond/dropout/div_grad/Shape*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*
Tshape0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
Ф
;training/Adam/gradients/dropout_1/cond/dropout/div_grad/NegNegdropout_1/cond/mul*-
_class#
!loc:@dropout_1/cond/dropout/div*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0

Atraining/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDiv_1RealDiv;training/Adam/gradients/dropout_1/cond/dropout/div_grad/Neg dropout_1/cond/dropout/keep_prob*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

Atraining/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDiv_2RealDivAtraining/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDiv_1 dropout_1/cond/dropout/keep_prob*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
Д
;training/Adam/gradients/dropout_1/cond/dropout/div_grad/mulMul?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ReshapeAtraining/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDiv_2*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
С
=training/Adam/gradients/dropout_1/cond/dropout/div_grad/Sum_1Sum;training/Adam/gradients/dropout_1/cond/dropout/div_grad/mulOtraining/Adam/gradients/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs:1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*
_output_shapes
:*
	keep_dims( *

Tidx0
Њ
Atraining/Adam/gradients/dropout_1/cond/dropout/div_grad/Reshape_1Reshape=training/Adam/gradients/dropout_1/cond/dropout/div_grad/Sum_1?training/Adam/gradients/dropout_1/cond/dropout/div_grad/Shape_1*
_output_shapes
: *
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*
Tshape0
З
5training/Adam/gradients/dropout_1/cond/mul_grad/ShapeShapedropout_1/cond/mul/Switch:1*
T0*%
_class
loc:@dropout_1/cond/mul*
out_type0*
_output_shapes
:
Ё
7training/Adam/gradients/dropout_1/cond/mul_grad/Shape_1Const*%
_class
loc:@dropout_1/cond/mul*
valueB *
dtype0*
_output_shapes
: 
В
Etraining/Adam/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/dropout_1/cond/mul_grad/Shape7training/Adam/gradients/dropout_1/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_1/cond/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ї
3training/Adam/gradients/dropout_1/cond/mul_grad/MulMul?training/Adam/gradients/dropout_1/cond/dropout/div_grad/Reshapedropout_1/cond/mul/y*
T0*%
_class
loc:@dropout_1/cond/mul*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

3training/Adam/gradients/dropout_1/cond/mul_grad/SumSum3training/Adam/gradients/dropout_1/cond/mul_grad/MulEtraining/Adam/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgs*%
_class
loc:@dropout_1/cond/mul*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ђ
7training/Adam/gradients/dropout_1/cond/mul_grad/ReshapeReshape3training/Adam/gradients/dropout_1/cond/mul_grad/Sum5training/Adam/gradients/dropout_1/cond/mul_grad/Shape*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0*%
_class
loc:@dropout_1/cond/mul*
Tshape0

5training/Adam/gradients/dropout_1/cond/mul_grad/Mul_1Muldropout_1/cond/mul/Switch:1?training/Adam/gradients/dropout_1/cond/dropout/div_grad/Reshape*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0*%
_class
loc:@dropout_1/cond/mul
Ѓ
5training/Adam/gradients/dropout_1/cond/mul_grad/Sum_1Sum5training/Adam/gradients/dropout_1/cond/mul_grad/Mul_1Gtraining/Adam/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@dropout_1/cond/mul*
_output_shapes
:*
	keep_dims( *

Tidx0

9training/Adam/gradients/dropout_1/cond/mul_grad/Reshape_1Reshape5training/Adam/gradients/dropout_1/cond/mul_grad/Sum_17training/Adam/gradients/dropout_1/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_1/cond/mul*
Tshape0*
_output_shapes
: 
м
 training/Adam/gradients/Switch_7Switchlstm_1/transpose_1dropout_1/cond/pred_id*%
_class
loc:@lstm_1/transpose_1*T
_output_shapesB
@:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ*
T0
Ж
"training/Adam/gradients/Identity_5Identity training/Adam/gradients/Switch_7*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0*%
_class
loc:@lstm_1/transpose_1
І
training/Adam/gradients/Shape_6Shape training/Adam/gradients/Switch_7*
T0*%
_class
loc:@lstm_1/transpose_1*
out_type0*
_output_shapes
:
Ж
%training/Adam/gradients/zeros_5/ConstConst#^training/Adam/gradients/Identity_5*
_output_shapes
: *%
_class
loc:@lstm_1/transpose_1*
valueB
 *    *
dtype0
ч
training/Adam/gradients/zeros_5Filltraining/Adam/gradients/Shape_6%training/Adam/gradients/zeros_5/Const*
T0*%
_class
loc:@lstm_1/transpose_1*

index_type0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

@training/Adam/gradients/dropout_1/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_57training/Adam/gradients/dropout_1/cond/mul_grad/Reshape*
T0*%
_class
loc:@lstm_1/transpose_1*
N*6
_output_shapes$
":џџџџџџџџџџџџџџџџџџ: 

training/Adam/gradients/AddN_3AddN>training/Adam/gradients/dropout_1/cond/Switch_1_grad/cond_grad@training/Adam/gradients/dropout_1/cond/mul/Switch_grad/cond_grad*
T0*%
_class
loc:@lstm_1/transpose_1*
N*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
Л
Atraining/Adam/gradients/lstm_1/transpose_1_grad/InvertPermutationInvertPermutationlstm_1/transpose_1/perm*
_output_shapes
:*
T0*%
_class
loc:@lstm_1/transpose_1

9training/Adam/gradients/lstm_1/transpose_1_grad/transpose	Transposetraining/Adam/gradients/AddN_3Atraining/Adam/gradients/lstm_1/transpose_1_grad/InvertPermutation*
T0*%
_class
loc:@lstm_1/transpose_1*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
Tperm0

jtraining/Adam/gradients/lstm_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lstm_1/TensorArraylstm_1/while/Exit_2*%
_class
loc:@lstm_1/TensorArray*#
sourcetraining/Adam/gradients*
_output_shapes

:: 
М
ftraining/Adam/gradients/lstm_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentitylstm_1/while/Exit_2k^training/Adam/gradients/lstm_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*%
_class
loc:@lstm_1/TensorArray*
_output_shapes
: 
ў
ptraining/Adam/gradients/lstm_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3jtraining/Adam/gradients/lstm_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3lstm_1/TensorArrayStack/range9training/Adam/gradients/lstm_1/transpose_1_grad/transposeftraining/Adam/gradients/lstm_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0*%
_class
loc:@lstm_1/TensorArray
v
"training/Adam/gradients/zeros_like	ZerosLikelstm_1/while/Exit_3*
T0*'
_output_shapes
:џџџџџџџџџ
x
$training/Adam/gradients/zeros_like_1	ZerosLikelstm_1/while/Exit_4*
T0*'
_output_shapes
:џџџџџџџџџ
э
7training/Adam/gradients/lstm_1/while/Exit_2_grad/b_exitEnterptraining/Adam/gradients/lstm_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
is_constant( *
_output_shapes
: *B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*&
_class
loc:@lstm_1/while/Exit_2*
parallel_iterations 
А
7training/Adam/gradients/lstm_1/while/Exit_3_grad/b_exitEnter"training/Adam/gradients/zeros_like*
is_constant( *'
_output_shapes
:џџџџџџџџџ*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*&
_class
loc:@lstm_1/while/Exit_3*
parallel_iterations 
В
7training/Adam/gradients/lstm_1/while/Exit_4_grad/b_exitEnter$training/Adam/gradients/zeros_like_1*
T0*&
_class
loc:@lstm_1/while/Exit_4*
parallel_iterations *
is_constant( *'
_output_shapes
:џџџџџџџџџ*B

frame_name42training/Adam/gradients/lstm_1/while/while_context

;training/Adam/gradients/lstm_1/while/Switch_2_grad/b_switchMerge7training/Adam/gradients/lstm_1/while/Exit_2_grad/b_exitBtraining/Adam/gradients/lstm_1/while/Switch_2_grad_1/NextIteration*
T0*'
_class
loc:@lstm_1/while/Merge_2*
N*
_output_shapes
: : 
Ї
;training/Adam/gradients/lstm_1/while/Switch_3_grad/b_switchMerge7training/Adam/gradients/lstm_1/while/Exit_3_grad/b_exitBtraining/Adam/gradients/lstm_1/while/Switch_3_grad_1/NextIteration*
T0*'
_class
loc:@lstm_1/while/Merge_3*
N*)
_output_shapes
:џџџџџџџџџ: 
Ї
;training/Adam/gradients/lstm_1/while/Switch_4_grad/b_switchMerge7training/Adam/gradients/lstm_1/while/Exit_4_grad/b_exitBtraining/Adam/gradients/lstm_1/while/Switch_4_grad_1/NextIteration*
T0*'
_class
loc:@lstm_1/while/Merge_4*
N*)
_output_shapes
:џџџџџџџџџ: 
ю
8training/Adam/gradients/lstm_1/while/Merge_2_grad/SwitchSwitch;training/Adam/gradients/lstm_1/while/Switch_2_grad/b_switch!training/Adam/gradients/b_count_2*
_output_shapes
: : *
T0*'
_class
loc:@lstm_1/while/Merge_2

8training/Adam/gradients/lstm_1/while/Merge_3_grad/SwitchSwitch;training/Adam/gradients/lstm_1/while/Switch_3_grad/b_switch!training/Adam/gradients/b_count_2*
T0*'
_class
loc:@lstm_1/while/Merge_3*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ

8training/Adam/gradients/lstm_1/while/Merge_4_grad/SwitchSwitch;training/Adam/gradients/lstm_1/while/Switch_4_grad/b_switch!training/Adam/gradients/b_count_2*
T0*'
_class
loc:@lstm_1/while/Merge_4*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
Т
6training/Adam/gradients/lstm_1/while/Enter_2_grad/ExitExit8training/Adam/gradients/lstm_1/while/Merge_2_grad/Switch*
_output_shapes
: *
T0*'
_class
loc:@lstm_1/while/Enter_2
г
6training/Adam/gradients/lstm_1/while/Enter_3_grad/ExitExit8training/Adam/gradients/lstm_1/while/Merge_3_grad/Switch*'
_output_shapes
:џџџџџџџџџ*
T0*'
_class
loc:@lstm_1/while/Enter_3
г
6training/Adam/gradients/lstm_1/while/Enter_4_grad/ExitExit8training/Adam/gradients/lstm_1/while/Merge_4_grad/Switch*'
_class
loc:@lstm_1/while/Enter_4*'
_output_shapes
:џџџџџџџџџ*
T0
Ё
otraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3utraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter:training/Adam/gradients/lstm_1/while/Merge_2_grad/Switch:1*%
_class
loc:@lstm_1/while/mul_5*#
sourcetraining/Adam/gradients*
_output_shapes

:: 
а
utraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlstm_1/TensorArray*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0
э
ktraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity:training/Adam/gradients/lstm_1/while/Merge_2_grad/Switch:1p^training/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
: 

_training/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3otraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3jtraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2ktraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*'
_output_shapes
:џџџџџџџџџ*%
_class
loc:@lstm_1/while/mul_5
ѕ
etraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*C
_class9
7loc:@lstm_1/while/Identity_1loc:@lstm_1/while/mul_5*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
л
etraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2etraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*C
_class9
7loc:@lstm_1/while/Identity_1loc:@lstm_1/while/mul_5*

stack_name *
_output_shapes
:*
	elem_type0
ћ
etraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnteretraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations 
ѓ
ktraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2etraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterlstm_1/while/Identity_1^training/Adam/gradients/Add*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
: *
swap_memory(*
T0
и
jtraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2ptraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
: *
	elem_type0

ptraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnteretraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations 
р
ftraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerG^training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV2G^training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV2k^training/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Q^training/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2_1Q^training/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopV2Q^training/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2_1Q^training/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopV2Q^training/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2_1Q^training/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2_1Q^training/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2_1Q^training/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPopV2a^training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopV2U^training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopV2c^training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopV2W^training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopV2[^training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopV2R^training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopV2c^training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopV2W^training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopV2[^training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopV2R^training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopV2Y^training/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopV2P^training/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopV2Q^training/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopV2?^training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPopV2Q^training/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2_1?^training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPopV2A^training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPopV2Q^training/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2_1?^training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPopV2A^training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPopV2Q^training/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopV2?^training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPopV2Q^training/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2_1?^training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV2A^training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPopV2O^training/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopV2=^training/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPopV2*%
_class
loc:@lstm_1/while/mul_5

.training/Adam/gradients/lstm_1/Tile_grad/ShapeShapelstm_1/ExpandDims*
_class
loc:@lstm_1/Tile*
out_type0*
_output_shapes
:*
T0
л
.training/Adam/gradients/lstm_1/Tile_grad/stackPacklstm_1/Tile/multiples.training/Adam/gradients/lstm_1/Tile_grad/Shape*
_class
loc:@lstm_1/Tile*

axis *
N*
_output_shapes

:*
T0
А
7training/Adam/gradients/lstm_1/Tile_grad/transpose/RankRank.training/Adam/gradients/lstm_1/Tile_grad/stack*
T0*
_class
loc:@lstm_1/Tile*
_output_shapes
: 

8training/Adam/gradients/lstm_1/Tile_grad/transpose/sub/yConst*
_class
loc:@lstm_1/Tile*
value	B :*
dtype0*
_output_shapes
: 
ё
6training/Adam/gradients/lstm_1/Tile_grad/transpose/subSub7training/Adam/gradients/lstm_1/Tile_grad/transpose/Rank8training/Adam/gradients/lstm_1/Tile_grad/transpose/sub/y*
T0*
_class
loc:@lstm_1/Tile*
_output_shapes
: 
 
>training/Adam/gradients/lstm_1/Tile_grad/transpose/Range/startConst*
_class
loc:@lstm_1/Tile*
value	B : *
dtype0*
_output_shapes
: 
 
>training/Adam/gradients/lstm_1/Tile_grad/transpose/Range/deltaConst*
_class
loc:@lstm_1/Tile*
value	B :*
dtype0*
_output_shapes
: 
Ы
8training/Adam/gradients/lstm_1/Tile_grad/transpose/RangeRange>training/Adam/gradients/lstm_1/Tile_grad/transpose/Range/start7training/Adam/gradients/lstm_1/Tile_grad/transpose/Rank>training/Adam/gradients/lstm_1/Tile_grad/transpose/Range/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
_class
loc:@lstm_1/Tile
џ
8training/Adam/gradients/lstm_1/Tile_grad/transpose/sub_1Sub6training/Adam/gradients/lstm_1/Tile_grad/transpose/sub8training/Adam/gradients/lstm_1/Tile_grad/transpose/Range*#
_output_shapes
:џџџџџџџџџ*
T0*
_class
loc:@lstm_1/Tile
џ
2training/Adam/gradients/lstm_1/Tile_grad/transpose	Transpose.training/Adam/gradients/lstm_1/Tile_grad/stack8training/Adam/gradients/lstm_1/Tile_grad/transpose/sub_1*
T0*
_class
loc:@lstm_1/Tile*
_output_shapes

:*
Tperm0
Љ
6training/Adam/gradients/lstm_1/Tile_grad/Reshape/shapeConst*
_class
loc:@lstm_1/Tile*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
њ
0training/Adam/gradients/lstm_1/Tile_grad/ReshapeReshape2training/Adam/gradients/lstm_1/Tile_grad/transpose6training/Adam/gradients/lstm_1/Tile_grad/Reshape/shape*
T0*
_class
loc:@lstm_1/Tile*
Tshape0*
_output_shapes
:

-training/Adam/gradients/lstm_1/Tile_grad/SizeConst*
dtype0*
_output_shapes
: *
_class
loc:@lstm_1/Tile*
value	B :

4training/Adam/gradients/lstm_1/Tile_grad/range/startConst*
_class
loc:@lstm_1/Tile*
value	B : *
dtype0*
_output_shapes
: 

4training/Adam/gradients/lstm_1/Tile_grad/range/deltaConst*
_class
loc:@lstm_1/Tile*
value	B :*
dtype0*
_output_shapes
: 

.training/Adam/gradients/lstm_1/Tile_grad/rangeRange4training/Adam/gradients/lstm_1/Tile_grad/range/start-training/Adam/gradients/lstm_1/Tile_grad/Size4training/Adam/gradients/lstm_1/Tile_grad/range/delta*
_class
loc:@lstm_1/Tile*
_output_shapes
:*

Tidx0
Њ
2training/Adam/gradients/lstm_1/Tile_grad/Reshape_1Reshape6training/Adam/gradients/lstm_1/while/Enter_3_grad/Exit0training/Adam/gradients/lstm_1/Tile_grad/Reshape*
T0*
_class
loc:@lstm_1/Tile*
Tshape0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ

,training/Adam/gradients/lstm_1/Tile_grad/SumSum2training/Adam/gradients/lstm_1/Tile_grad/Reshape_1.training/Adam/gradients/lstm_1/Tile_grad/range*'
_output_shapes
:џџџџџџџџџ*
	keep_dims( *

Tidx0*
T0*
_class
loc:@lstm_1/Tile
Ѓ
0training/Adam/gradients/lstm_1/Tile_1_grad/ShapeShapelstm_1/ExpandDims*
_output_shapes
:*
T0* 
_class
loc:@lstm_1/Tile_1*
out_type0
у
0training/Adam/gradients/lstm_1/Tile_1_grad/stackPacklstm_1/Tile_1/multiples0training/Adam/gradients/lstm_1/Tile_1_grad/Shape* 
_class
loc:@lstm_1/Tile_1*

axis *
N*
_output_shapes

:*
T0
Ж
9training/Adam/gradients/lstm_1/Tile_1_grad/transpose/RankRank0training/Adam/gradients/lstm_1/Tile_1_grad/stack*
_output_shapes
: *
T0* 
_class
loc:@lstm_1/Tile_1

:training/Adam/gradients/lstm_1/Tile_1_grad/transpose/sub/yConst* 
_class
loc:@lstm_1/Tile_1*
value	B :*
dtype0*
_output_shapes
: 
љ
8training/Adam/gradients/lstm_1/Tile_1_grad/transpose/subSub9training/Adam/gradients/lstm_1/Tile_1_grad/transpose/Rank:training/Adam/gradients/lstm_1/Tile_1_grad/transpose/sub/y* 
_class
loc:@lstm_1/Tile_1*
_output_shapes
: *
T0
Є
@training/Adam/gradients/lstm_1/Tile_1_grad/transpose/Range/startConst* 
_class
loc:@lstm_1/Tile_1*
value	B : *
dtype0*
_output_shapes
: 
Є
@training/Adam/gradients/lstm_1/Tile_1_grad/transpose/Range/deltaConst*
_output_shapes
: * 
_class
loc:@lstm_1/Tile_1*
value	B :*
dtype0
е
:training/Adam/gradients/lstm_1/Tile_1_grad/transpose/RangeRange@training/Adam/gradients/lstm_1/Tile_1_grad/transpose/Range/start9training/Adam/gradients/lstm_1/Tile_1_grad/transpose/Rank@training/Adam/gradients/lstm_1/Tile_1_grad/transpose/Range/delta* 
_class
loc:@lstm_1/Tile_1*#
_output_shapes
:џџџџџџџџџ*

Tidx0

:training/Adam/gradients/lstm_1/Tile_1_grad/transpose/sub_1Sub8training/Adam/gradients/lstm_1/Tile_1_grad/transpose/sub:training/Adam/gradients/lstm_1/Tile_1_grad/transpose/Range*
T0* 
_class
loc:@lstm_1/Tile_1*#
_output_shapes
:џџџџџџџџџ

4training/Adam/gradients/lstm_1/Tile_1_grad/transpose	Transpose0training/Adam/gradients/lstm_1/Tile_1_grad/stack:training/Adam/gradients/lstm_1/Tile_1_grad/transpose/sub_1*
T0* 
_class
loc:@lstm_1/Tile_1*
_output_shapes

:*
Tperm0
­
8training/Adam/gradients/lstm_1/Tile_1_grad/Reshape/shapeConst* 
_class
loc:@lstm_1/Tile_1*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:

2training/Adam/gradients/lstm_1/Tile_1_grad/ReshapeReshape4training/Adam/gradients/lstm_1/Tile_1_grad/transpose8training/Adam/gradients/lstm_1/Tile_1_grad/Reshape/shape*
T0* 
_class
loc:@lstm_1/Tile_1*
Tshape0*
_output_shapes
:

/training/Adam/gradients/lstm_1/Tile_1_grad/SizeConst*
dtype0*
_output_shapes
: * 
_class
loc:@lstm_1/Tile_1*
value	B :

6training/Adam/gradients/lstm_1/Tile_1_grad/range/startConst* 
_class
loc:@lstm_1/Tile_1*
value	B : *
dtype0*
_output_shapes
: 

6training/Adam/gradients/lstm_1/Tile_1_grad/range/deltaConst* 
_class
loc:@lstm_1/Tile_1*
value	B :*
dtype0*
_output_shapes
: 
Є
0training/Adam/gradients/lstm_1/Tile_1_grad/rangeRange6training/Adam/gradients/lstm_1/Tile_1_grad/range/start/training/Adam/gradients/lstm_1/Tile_1_grad/Size6training/Adam/gradients/lstm_1/Tile_1_grad/range/delta*
_output_shapes
:*

Tidx0* 
_class
loc:@lstm_1/Tile_1
А
4training/Adam/gradients/lstm_1/Tile_1_grad/Reshape_1Reshape6training/Adam/gradients/lstm_1/while/Enter_4_grad/Exit2training/Adam/gradients/lstm_1/Tile_1_grad/Reshape*
T0* 
_class
loc:@lstm_1/Tile_1*
Tshape0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ

.training/Adam/gradients/lstm_1/Tile_1_grad/SumSum4training/Adam/gradients/lstm_1/Tile_1_grad/Reshape_10training/Adam/gradients/lstm_1/Tile_1_grad/range*
	keep_dims( *

Tidx0*
T0* 
_class
loc:@lstm_1/Tile_1*'
_output_shapes
:џџџџџџџџџ
Ї
training/Adam/gradients/AddN_4AddN:training/Adam/gradients/lstm_1/while/Merge_3_grad/Switch:1_training/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
T0*'
_class
loc:@lstm_1/while/Merge_3*
N*'
_output_shapes
:џџџџџџџџџ
И
5training/Adam/gradients/lstm_1/while/mul_5_grad/ShapeShapelstm_1/while/clip_by_value_2*
T0*%
_class
loc:@lstm_1/while/mul_5*
out_type0*
_output_shapes
:
Б
7training/Adam/gradients/lstm_1/while/mul_5_grad/Shape_1Shapelstm_1/while/Tanh_1*%
_class
loc:@lstm_1/while/mul_5*
out_type0*
_output_shapes
:*
T0
ш
Etraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2Rtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/mul_5*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/mul_5*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/mul_5
Ч
Ktraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0
с
Qtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/mul_5_grad/Shape^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
:*
swap_memory(
Ј
Ptraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/mul_5
ъ
Vtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
П
Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Const_1Const*%
_class
loc:@lstm_1/while/mul_5*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Const_1*%
_class
loc:@lstm_1/while/mul_5*

stack_name *
_output_shapes
:*
	elem_type0
Ы
Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc_1*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations 
ч
Straining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/mul_5_grad/Shape_1^training/Adam/gradients/Add*
_output_shapes
:*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/mul_5
Ќ
Rtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*
	elem_type0*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
:
ю
Xtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterMtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc_1*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
ѓ
3training/Adam/gradients/lstm_1/while/mul_5_grad/MulMultraining/Adam/gradients/AddN_4>training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/mul_5
Х
9training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/ConstConst*?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_5*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
џ
9training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/f_accStackV29training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/Const*?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_5*

stack_name *
_output_shapes
:*
	elem_type0
Ѓ
9training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/EnterEnter9training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations *
is_constant(
Ј
?training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPushV2StackPushV29training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/Enterlstm_1/while/Tanh_1^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_5*'
_output_shapes
:џџџџџџџџџ*
swap_memory(

>training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV2
StackPopV2Dtraining/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*
	elem_type0*%
_class
loc:@lstm_1/while/mul_5*'
_output_shapes
:џџџџџџџџџ
Ц
Dtraining/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV2/EnterEnter9training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5

3training/Adam/gradients/lstm_1/while/mul_5_grad/SumSum3training/Adam/gradients/lstm_1/while/mul_5_grad/MulEtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs*
T0*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
:*
	keep_dims( *

Tidx0
А
7training/Adam/gradients/lstm_1/while/mul_5_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/mul_5_grad/SumPtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_5*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ї
5training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1Mul@training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPopV2training/Adam/gradients/AddN_4*
T0*%
_class
loc:@lstm_1/while/mul_5*'
_output_shapes
:џџџџџџџџџ
а
;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_5*
valueB :
џџџџџџџџџ

;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/f_accStackV2;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/Const*
_output_shapes
:*
	elem_type0*H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_5*

stack_name 
Ї
;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/EnterEnter;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations 
Е
Atraining/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPushV2StackPushV2;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/Enterlstm_1/while/clip_by_value_2^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_5*'
_output_shapes
:џџџџџџџџџ*
swap_memory(

@training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPopV2
StackPopV2Ftraining/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPopV2/Enter^training/Adam/gradients/Sub*'
_output_shapes
:џџџџџџџџџ*
	elem_type0*%
_class
loc:@lstm_1/while/mul_5
Ъ
Ftraining/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPopV2/EnterEnter;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/f_acc*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations *
is_constant(
Ѓ
5training/Adam/gradients/lstm_1/while/mul_5_grad/Sum_1Sum5training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1Gtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/mul_5
Ж
9training/Adam/gradients/lstm_1/while/mul_5_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/mul_5_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2_1*%
_class
loc:@lstm_1/while/mul_5*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
п
training/Adam/gradients/AddN_5AddN,training/Adam/gradients/lstm_1/Tile_grad/Sum.training/Adam/gradients/lstm_1/Tile_1_grad/Sum*
N*'
_output_shapes
:џџџџџџџџџ*
T0*
_class
loc:@lstm_1/Tile
Є
4training/Adam/gradients/lstm_1/ExpandDims_grad/ShapeShape
lstm_1/Sum*
T0*$
_class
loc:@lstm_1/ExpandDims*
out_type0*
_output_shapes
:
љ
6training/Adam/gradients/lstm_1/ExpandDims_grad/ReshapeReshapetraining/Adam/gradients/AddN_54training/Adam/gradients/lstm_1/ExpandDims_grad/Shape*#
_output_shapes
:џџџџџџџџџ*
T0*$
_class
loc:@lstm_1/ExpandDims*
Tshape0
д
?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/ShapeShape$lstm_1/while/clip_by_value_2/Minimum*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
out_type0*
_output_shapes
:
г
Atraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Shape_1Const^training/Adam/gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
valueB *
dtype0*
_output_shapes
: 
щ
Atraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Shape_2Shape7training/Adam/gradients/lstm_1/while/mul_5_grad/Reshape*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
out_type0*
_output_shapes
:
й
Etraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/zeros/ConstConst^training/Adam/gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
valueB
 *    *
dtype0*
_output_shapes
: 
Ц
?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/zerosFillAtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Shape_2Etraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/zeros/Const*'
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*

index_type0
м
Ftraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqualGreaterEqualQtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopV2Ntraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/Const_1*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*'
_output_shapes
:џџџџџџџџџ
ѓ
Ltraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/ConstConst*Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Р
Ltraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_accStackV2Ltraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/Const*Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*

stack_name *
_output_shapes
:*
	elem_type0
г
Ltraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/EnterEnterLtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
щ
Rtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPushV2StackPushV2Ltraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/Enter$lstm_1/while/clip_by_value_2/Minimum^training/Adam/gradients/Add*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*'
_output_shapes
:џџџџџџџџџ*
swap_memory(*
T0
С
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopV2
StackPopV2Wtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopV2/Enter^training/Adam/gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
і
Wtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopV2/EnterEnterLtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
parallel_iterations 
т
Ntraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/Const_1Const^training/Adam/gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
valueB
 *    *
dtype0*
_output_shapes
: 
ѕ
Otraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgsBroadcastGradientArgsZtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopV2Atraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Shape_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
б
Utraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/ConstConst*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ї
Utraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_accStackV2Utraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/Const*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*

stack_name *
_output_shapes
:*
	elem_type0
х
Utraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/EnterEnterUtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_acc*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0

[training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2Utraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/Enter?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Shape^training/Adam/gradients/Add*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
_output_shapes
:*
swap_memory(
Ц
Ztraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2`training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
_output_shapes
:*
	elem_type0

`training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterUtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
parallel_iterations *
is_constant(
я
@training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/SelectSelectFtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual7training/Adam/gradients/lstm_1/while/mul_5_grad/Reshape?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/zeros*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*'
_output_shapes
:џџџџџџџџџ
ё
Btraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Select_1SelectFtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/zeros7training/Adam/gradients/lstm_1/while/mul_5_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2
Ш
=training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/SumSum@training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/SelectOtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
и
Atraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/ReshapeReshape=training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/SumZtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
Tshape0
Ю
?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Sum_1SumBtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Select_1Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs:1*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
_output_shapes
:*
	keep_dims( *

Tidx0
В
Ctraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Reshape_1Reshape?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Sum_1Atraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Shape_1*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
Tshape0*
_output_shapes
: 

9training/Adam/gradients/lstm_1/while/Tanh_1_grad/TanhGradTanhGrad>training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV29training/Adam/gradients/lstm_1/while/mul_5_grad/Reshape_1*
T0*&
_class
loc:@lstm_1/while/Tanh_1*'
_output_shapes
:џџџџџџџџџ
й
Btraining/Adam/gradients/lstm_1/while/Switch_2_grad_1/NextIterationNextIteration:training/Adam/gradients/lstm_1/while/Merge_2_grad/Switch:1*
T0*'
_class
loc:@lstm_1/while/Merge_2*
_output_shapes
: 

-training/Adam/gradients/lstm_1/Sum_grad/ShapeShapelstm_1/zeros_like*
_output_shapes
:*
T0*
_class
loc:@lstm_1/Sum*
out_type0

,training/Adam/gradients/lstm_1/Sum_grad/SizeConst*
_class
loc:@lstm_1/Sum*
value	B :*
dtype0*
_output_shapes
: 
Т
+training/Adam/gradients/lstm_1/Sum_grad/addAddlstm_1/Sum/reduction_indices,training/Adam/gradients/lstm_1/Sum_grad/Size*
_output_shapes
:*
T0*
_class
loc:@lstm_1/Sum
ж
+training/Adam/gradients/lstm_1/Sum_grad/modFloorMod+training/Adam/gradients/lstm_1/Sum_grad/add,training/Adam/gradients/lstm_1/Sum_grad/Size*
T0*
_class
loc:@lstm_1/Sum*
_output_shapes
:

/training/Adam/gradients/lstm_1/Sum_grad/Shape_1Const*
_class
loc:@lstm_1/Sum*
valueB:*
dtype0*
_output_shapes
:

3training/Adam/gradients/lstm_1/Sum_grad/range/startConst*
_class
loc:@lstm_1/Sum*
value	B : *
dtype0*
_output_shapes
: 

3training/Adam/gradients/lstm_1/Sum_grad/range/deltaConst*
_class
loc:@lstm_1/Sum*
value	B :*
dtype0*
_output_shapes
: 

-training/Adam/gradients/lstm_1/Sum_grad/rangeRange3training/Adam/gradients/lstm_1/Sum_grad/range/start,training/Adam/gradients/lstm_1/Sum_grad/Size3training/Adam/gradients/lstm_1/Sum_grad/range/delta*
_output_shapes
:*

Tidx0*
_class
loc:@lstm_1/Sum

2training/Adam/gradients/lstm_1/Sum_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
_class
loc:@lstm_1/Sum*
value	B :
я
,training/Adam/gradients/lstm_1/Sum_grad/FillFill/training/Adam/gradients/lstm_1/Sum_grad/Shape_12training/Adam/gradients/lstm_1/Sum_grad/Fill/value*
_class
loc:@lstm_1/Sum*

index_type0*
_output_shapes
:*
T0
Ь
5training/Adam/gradients/lstm_1/Sum_grad/DynamicStitchDynamicStitch-training/Adam/gradients/lstm_1/Sum_grad/range+training/Adam/gradients/lstm_1/Sum_grad/mod-training/Adam/gradients/lstm_1/Sum_grad/Shape,training/Adam/gradients/lstm_1/Sum_grad/Fill*
T0*
_class
loc:@lstm_1/Sum*
N*
_output_shapes
:

1training/Adam/gradients/lstm_1/Sum_grad/Maximum/yConst*
_class
loc:@lstm_1/Sum*
value	B :*
dtype0*
_output_shapes
: 
ш
/training/Adam/gradients/lstm_1/Sum_grad/MaximumMaximum5training/Adam/gradients/lstm_1/Sum_grad/DynamicStitch1training/Adam/gradients/lstm_1/Sum_grad/Maximum/y*
T0*
_class
loc:@lstm_1/Sum*
_output_shapes
:
р
0training/Adam/gradients/lstm_1/Sum_grad/floordivFloorDiv-training/Adam/gradients/lstm_1/Sum_grad/Shape/training/Adam/gradients/lstm_1/Sum_grad/Maximum*
T0*
_class
loc:@lstm_1/Sum*
_output_shapes
:

/training/Adam/gradients/lstm_1/Sum_grad/ReshapeReshape6training/Adam/gradients/lstm_1/ExpandDims_grad/Reshape5training/Adam/gradients/lstm_1/Sum_grad/DynamicStitch*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*
_class
loc:@lstm_1/Sum*
Tshape0
џ
,training/Adam/gradients/lstm_1/Sum_grad/TileTile/training/Adam/gradients/lstm_1/Sum_grad/Reshape0training/Adam/gradients/lstm_1/Sum_grad/floordiv*
T0*
_class
loc:@lstm_1/Sum*,
_output_shapes
:џџџџџџџџџЅ@*

Tmultiples0
в
Gtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ShapeShapelstm_1/while/add_8*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
out_type0*
_output_shapes
:
у
Itraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_1Const^training/Adam/gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB *
dtype0*
_output_shapes
: 

Itraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_2ShapeAtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Reshape*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
out_type0*
_output_shapes
:
щ
Mtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros/ConstConst^training/Adam/gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB
 *    *
dtype0*
_output_shapes
: 
ц
Gtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zerosFillItraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_2Mtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros/Const*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*

index_type0*'
_output_shapes
:џџџџџџџџџ
№
Ktraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual	LessEqualVtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopV2Straining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/Const_1*'
_output_shapes
:џџџџџџџџџ*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum
ю
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/ConstConst*P
_classF
Dloc:@lstm_1/while/add_8)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Р
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_accStackV2Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/Const*
_output_shapes
:*
	elem_type0*P
_classF
Dloc:@lstm_1/while/add_8)loc:@lstm_1/while/clip_by_value_2/Minimum*

stack_name 
х
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/EnterEnterQtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0
щ
Wtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPushV2StackPushV2Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/Enterlstm_1/while/add_8^training/Adam/gradients/Add*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*'
_output_shapes
:џџџџџџџџџ*
swap_memory(
г
Vtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopV2
StackPopV2\training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopV2/Enter^training/Adam/gradients/Sub*'
_output_shapes
:џџџџџџџџџ*
	elem_type0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum

\training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopV2/EnterEnterQtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
parallel_iterations 
я
Straining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/Const_1Const^training/Adam/gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Wtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsbtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopV2Itraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_1*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
с
]training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/ConstConst*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
П
]training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_accStackV2]training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/Const*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*

stack_name *
_output_shapes
:*
	elem_type0
§
]training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/EnterEnter]training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
parallel_iterations 
Љ
ctraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPushV2StackPushV2]training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/EnterGtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape^training/Adam/gradients/Add*
_output_shapes
:*
swap_memory(*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum
о
btraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopV2
StackPopV2htraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
	elem_type0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
_output_shapes
:
 
htraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopV2/EnterEnter]training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
parallel_iterations 

Htraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SelectSelectKtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqualAtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/ReshapeGtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*'
_output_shapes
:џџџџџџџџџ

Jtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Select_1SelectKtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqualGtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zerosAtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Reshape*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*'
_output_shapes
:џџџџџџџџџ
ш
Etraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SumSumHtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SelectWtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
_output_shapes
:
ј
Itraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ReshapeReshapeEtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Sumbtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
Tshape0
ю
Gtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Sum_1SumJtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Select_1Ytraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum
в
Ktraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Reshape_1ReshapeGtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Sum_1Itraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_1*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
Tshape0*
_output_shapes
: *
T0

training/Adam/gradients/AddN_6AddN:training/Adam/gradients/lstm_1/while/Merge_4_grad/Switch:19training/Adam/gradients/lstm_1/while/Tanh_1_grad/TanhGrad*
T0*'
_class
loc:@lstm_1/while/Merge_4*
N*'
_output_shapes
:џџџџџџџџџ
Ў
5training/Adam/gradients/lstm_1/while/add_6_grad/ShapeShapelstm_1/while/mul_2*
_output_shapes
:*
T0*%
_class
loc:@lstm_1/while/add_6*
out_type0
А
7training/Adam/gradients/lstm_1/while/add_6_grad/Shape_1Shapelstm_1/while/mul_3*
_output_shapes
:*
T0*%
_class
loc:@lstm_1/while/add_6*
out_type0
ш
Etraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2Rtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/add_6*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/add_6*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/add_6*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc*
T0*%
_class
loc:@lstm_1/while/add_6*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
с
Qtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/add_6_grad/Shape^training/Adam/gradients/Add*
_output_shapes
:*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/add_6
Ј
Ptraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_6*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc*
T0*%
_class
loc:@lstm_1/while/add_6*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
П
Mtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/Const_1Const*%
_class
loc:@lstm_1/while/add_6*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Mtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/Const_1*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/add_6*

stack_name 
Ы
Mtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_6*
parallel_iterations *
is_constant(
ч
Straining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/add_6_grad/Shape_1^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/add_6*
_output_shapes
:*
swap_memory(
Ќ
Rtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/add_6
ю
Xtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterMtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1*
T0*%
_class
loc:@lstm_1/while/add_6*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context

3training/Adam/gradients/lstm_1/while/add_6_grad/SumSumtraining/Adam/gradients/AddN_6Etraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs*
T0*%
_class
loc:@lstm_1/while/add_6*
_output_shapes
:*
	keep_dims( *

Tidx0
А
7training/Adam/gradients/lstm_1/while/add_6_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_6_grad/SumPtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/add_6*
Tshape0*'
_output_shapes
:џџџџџџџџџ

5training/Adam/gradients/lstm_1/while/add_6_grad/Sum_1Sumtraining/Adam/gradients/AddN_6Gtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_6*
_output_shapes
:
Ж
9training/Adam/gradients/lstm_1/while/add_6_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_6_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/add_6*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ў
5training/Adam/gradients/lstm_1/while/add_8_grad/ShapeShapelstm_1/while/mul_4*
T0*%
_class
loc:@lstm_1/while/add_8*
out_type0*
_output_shapes
:
П
7training/Adam/gradients/lstm_1/while/add_8_grad/Shape_1Const^training/Adam/gradients/Sub*
dtype0*
_output_shapes
: *%
_class
loc:@lstm_1/while/add_8*
valueB 
Э
Etraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPopV27training/Adam/gradients/lstm_1/while/add_8_grad/Shape_1*
T0*%
_class
loc:@lstm_1/while/add_8*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/add_8*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/add_8*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_8*
parallel_iterations 
с
Qtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/add_8_grad/Shape^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/add_8*
_output_shapes
:*
swap_memory(
Ј
Ptraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_8*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/f_acc*
T0*%
_class
loc:@lstm_1/while/add_8*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
Г
3training/Adam/gradients/lstm_1/while/add_8_grad/SumSumItraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ReshapeEtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs*
T0*%
_class
loc:@lstm_1/while/add_8*
_output_shapes
:*
	keep_dims( *

Tidx0
А
7training/Adam/gradients/lstm_1/while/add_8_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_8_grad/SumPtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/add_8*
Tshape0*'
_output_shapes
:џџџџџџџџџ
З
5training/Adam/gradients/lstm_1/while/add_8_grad/Sum_1SumItraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ReshapeGtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@lstm_1/while/add_8*
_output_shapes
:*
	keep_dims( *

Tidx0

9training/Adam/gradients/lstm_1/while/add_8_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_8_grad/Sum_17training/Adam/gradients/lstm_1/while/add_8_grad/Shape_1*
T0*%
_class
loc:@lstm_1/while/add_8*
Tshape0*
_output_shapes
: 
И
5training/Adam/gradients/lstm_1/while/mul_2_grad/ShapeShapelstm_1/while/clip_by_value_1*%
_class
loc:@lstm_1/while/mul_2*
out_type0*
_output_shapes
:*
T0
Е
7training/Adam/gradients/lstm_1/while/mul_2_grad/Shape_1Shapelstm_1/while/Identity_4*
T0*%
_class
loc:@lstm_1/while/mul_2*
out_type0*
_output_shapes
:
ш
Etraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2Rtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/mul_2*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/mul_2*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/mul_2*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
с
Qtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/mul_2_grad/Shape^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_2*
_output_shapes
:*
swap_memory(
Ј
Ptraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_2*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations 
П
Mtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/Const_1Const*%
_class
loc:@lstm_1/while/mul_2*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Mtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/Const_1*

stack_name *
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/mul_2
Ы
Mtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc_1*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations 
ч
Straining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/mul_2_grad/Shape_1^training/Adam/gradients/Add*%
_class
loc:@lstm_1/while/mul_2*
_output_shapes
:*
swap_memory(*
T0
Ќ
Rtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*
	elem_type0*%
_class
loc:@lstm_1/while/mul_2*
_output_shapes
:
ю
Xtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterMtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context

3training/Adam/gradients/lstm_1/while/mul_2_grad/MulMul7training/Adam/gradients/lstm_1/while/add_6_grad/Reshape>training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_2*'
_output_shapes
:џџџџџџџџџ
Щ
9training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/ConstConst*C
_class9
7loc:@lstm_1/while/Identity_4loc:@lstm_1/while/mul_2*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

9training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/f_accStackV29training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/Const*C
_class9
7loc:@lstm_1/while/Identity_4loc:@lstm_1/while/mul_2*

stack_name *
_output_shapes
:*
	elem_type0
Ѓ
9training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/EnterEnter9training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations *
is_constant(
Ќ
?training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPushV2StackPushV29training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/Enterlstm_1/while/Identity_4^training/Adam/gradients/Add*'
_output_shapes
:џџџџџџџџџ*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/mul_2

>training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPopV2
StackPopV2Dtraining/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*'
_output_shapes
:џџџџџџџџџ*
	elem_type0*%
_class
loc:@lstm_1/while/mul_2
Ц
Dtraining/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPopV2/EnterEnter9training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations 

3training/Adam/gradients/lstm_1/while/mul_2_grad/SumSum3training/Adam/gradients/lstm_1/while/mul_2_grad/MulEtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/mul_2
А
7training/Adam/gradients/lstm_1/while/mul_2_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/mul_2_grad/SumPtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2*%
_class
loc:@lstm_1/while/mul_2*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0

5training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1Mul@training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPopV27training/Adam/gradients/lstm_1/while/add_6_grad/Reshape*
T0*%
_class
loc:@lstm_1/while/mul_2*'
_output_shapes
:џџџџџџџџџ
а
;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/ConstConst*H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_2*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/f_accStackV2;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/Const*H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_2*

stack_name *
_output_shapes
:*
	elem_type0
Ї
;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/EnterEnter;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
Е
Atraining/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPushV2StackPushV2;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/Enterlstm_1/while/clip_by_value_1^training/Adam/gradients/Add*%
_class
loc:@lstm_1/while/mul_2*'
_output_shapes
:џџџџџџџџџ*
swap_memory(*
T0

@training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPopV2
StackPopV2Ftraining/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_2*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
Ъ
Ftraining/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPopV2/EnterEnter;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
Ѓ
5training/Adam/gradients/lstm_1/while/mul_2_grad/Sum_1Sum5training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1Gtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/mul_2*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ж
9training/Adam/gradients/lstm_1/while/mul_2_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/mul_2_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/mul_2*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ж
5training/Adam/gradients/lstm_1/while/mul_3_grad/ShapeShapelstm_1/while/clip_by_value*
T0*%
_class
loc:@lstm_1/while/mul_3*
out_type0*
_output_shapes
:
Џ
7training/Adam/gradients/lstm_1/while/mul_3_grad/Shape_1Shapelstm_1/while/Tanh*%
_class
loc:@lstm_1/while/mul_3*
out_type0*
_output_shapes
:*
T0
ш
Etraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2Rtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/mul_3*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/mul_3*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/mul_3
Ч
Ktraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations 
с
Qtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/mul_3_grad/Shape^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_3*
_output_shapes
:*
swap_memory(
Ј
Ptraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_3*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations 
П
Mtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Const_1Const*%
_class
loc:@lstm_1/while/mul_3*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Mtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Const_1*%
_class
loc:@lstm_1/while/mul_3*

stack_name *
_output_shapes
:*
	elem_type0
Ы
Mtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc_1*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0
ч
Straining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/mul_3_grad/Shape_1^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_3*
_output_shapes
:*
swap_memory(
Ќ
Rtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_3*
_output_shapes
:*
	elem_type0
ю
Xtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterMtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc_1*
T0*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context

3training/Adam/gradients/lstm_1/while/mul_3_grad/MulMul9training/Adam/gradients/lstm_1/while/add_6_grad/Reshape_1>training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/mul_3
У
9training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/ConstConst*=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_3*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
§
9training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/f_accStackV29training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/Const*=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_3*

stack_name *
_output_shapes
:*
	elem_type0
Ѓ
9training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/EnterEnter9training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
І
?training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPushV2StackPushV29training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/Enterlstm_1/while/Tanh^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_3*'
_output_shapes
:џџџџџџџџџ*
swap_memory(

>training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPopV2
StackPopV2Dtraining/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*'
_output_shapes
:џџџџџџџџџ*
	elem_type0*%
_class
loc:@lstm_1/while/mul_3
Ц
Dtraining/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPopV2/EnterEnter9training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations 

3training/Adam/gradients/lstm_1/while/mul_3_grad/SumSum3training/Adam/gradients/lstm_1/while/mul_3_grad/MulEtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs*
T0*%
_class
loc:@lstm_1/while/mul_3*
_output_shapes
:*
	keep_dims( *

Tidx0
А
7training/Adam/gradients/lstm_1/while/mul_3_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/mul_3_grad/SumPtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_3*
Tshape0*'
_output_shapes
:џџџџџџџџџ

5training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1Mul@training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPopV29training/Adam/gradients/lstm_1/while/add_6_grad/Reshape_1*
T0*%
_class
loc:@lstm_1/while/mul_3*'
_output_shapes
:џџџџџџџџџ
Ю
;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_3*
valueB :
џџџџџџџџџ

;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/f_accStackV2;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/Const*
_output_shapes
:*
	elem_type0*F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_3*

stack_name 
Ї
;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/EnterEnter;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
Г
Atraining/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPushV2StackPushV2;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/Enterlstm_1/while/clip_by_value^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_3*'
_output_shapes
:џџџџџџџџџ*
swap_memory(

@training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPopV2
StackPopV2Ftraining/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_3*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
Ъ
Ftraining/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPopV2/EnterEnter;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
Ѓ
5training/Adam/gradients/lstm_1/while/mul_3_grad/Sum_1Sum5training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1Gtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@lstm_1/while/mul_3*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
9training/Adam/gradients/lstm_1/while/mul_3_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/mul_3_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/mul_3*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Н
5training/Adam/gradients/lstm_1/while/mul_4_grad/ShapeConst^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_4*
valueB *
dtype0*
_output_shapes
: 
А
7training/Adam/gradients/lstm_1/while/mul_4_grad/Shape_1Shapelstm_1/while/add_7*
T0*%
_class
loc:@lstm_1/while/mul_4*
out_type0*
_output_shapes
:
Ы
Etraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/lstm_1/while/mul_4_grad/ShapePtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_4*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/mul_4*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/mul_4*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_4*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
у
Qtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/Enter7training/Adam/gradients/lstm_1/while/mul_4_grad/Shape_1^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_4*
_output_shapes
:*
swap_memory(
Ј
Ptraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/mul_4
ъ
Vtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_4

3training/Adam/gradients/lstm_1/while/mul_4_grad/MulMul7training/Adam/gradients/lstm_1/while/add_8_grad/Reshape>training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_4*'
_output_shapes
:џџџџџџџџџ
Ф
9training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/ConstConst*>
_class4
2loc:@lstm_1/while/add_7loc:@lstm_1/while/mul_4*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
ў
9training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/f_accStackV29training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/Const*

stack_name *
_output_shapes
:*
	elem_type0*>
_class4
2loc:@lstm_1/while/add_7loc:@lstm_1/while/mul_4
Ѓ
9training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/EnterEnter9training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/f_acc*%
_class
loc:@lstm_1/while/mul_4*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0
Ї
?training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPushV2StackPushV29training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/Enterlstm_1/while/add_7^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_4*'
_output_shapes
:џџџџџџџџџ*
swap_memory(

>training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPopV2
StackPopV2Dtraining/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_4*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
Ц
Dtraining/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPopV2/EnterEnter9training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_4*
parallel_iterations 

3training/Adam/gradients/lstm_1/while/mul_4_grad/SumSum3training/Adam/gradients/lstm_1/while/mul_4_grad/MulEtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs*
T0*%
_class
loc:@lstm_1/while/mul_4*
_output_shapes
:*
	keep_dims( *

Tidx0

7training/Adam/gradients/lstm_1/while/mul_4_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/mul_4_grad/Sum5training/Adam/gradients/lstm_1/while/mul_4_grad/Shape*
T0*%
_class
loc:@lstm_1/while/mul_4*
Tshape0*
_output_shapes
: 

5training/Adam/gradients/lstm_1/while/mul_4_grad/Mul_1Mul;training/Adam/gradients/lstm_1/while/mul_4_grad/Mul_1/Const7training/Adam/gradients/lstm_1/while/add_8_grad/Reshape*
T0*%
_class
loc:@lstm_1/while/mul_4*'
_output_shapes
:џџџџџџџџџ
Х
;training/Adam/gradients/lstm_1/while/mul_4_grad/Mul_1/ConstConst^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_4*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
Ѓ
5training/Adam/gradients/lstm_1/while/mul_4_grad/Sum_1Sum5training/Adam/gradients/lstm_1/while/mul_4_grad/Mul_1Gtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@lstm_1/while/mul_4*
_output_shapes
:*
	keep_dims( *

Tidx0
Д
9training/Adam/gradients/lstm_1/while/mul_4_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/mul_4_grad/Sum_1Ptraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_4*
Tshape0*'
_output_shapes
:џџџџџџџџџ
д
?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/ShapeShape$lstm_1/while/clip_by_value_1/Minimum*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
out_type0*
_output_shapes
:*
T0
г
Atraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Shape_1Const^training/Adam/gradients/Sub*
_output_shapes
: */
_class%
#!loc:@lstm_1/while/clip_by_value_1*
valueB *
dtype0
щ
Atraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Shape_2Shape7training/Adam/gradients/lstm_1/while/mul_2_grad/Reshape*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
out_type0*
_output_shapes
:
й
Etraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/zeros/ConstConst^training/Adam/gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
valueB
 *    *
dtype0*
_output_shapes
: 
Ц
?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/zerosFillAtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Shape_2Etraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/zeros/Const*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*

index_type0*'
_output_shapes
:џџџџџџџџџ
м
Ftraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqualGreaterEqualQtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopV2Ntraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/Const_1*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*'
_output_shapes
:џџџџџџџџџ
ѓ
Ltraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/ConstConst*Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Р
Ltraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_accStackV2Ltraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/Const*Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*

stack_name *
_output_shapes
:*
	elem_type0
г
Ltraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/EnterEnterLtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
parallel_iterations 
щ
Rtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPushV2StackPushV2Ltraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/Enter$lstm_1/while/clip_by_value_1/Minimum^training/Adam/gradients/Add*'
_output_shapes
:џџџџџџџџџ*
swap_memory(*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1
С
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopV2
StackPopV2Wtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopV2/Enter^training/Adam/gradients/Sub*'
_output_shapes
:џџџџџџџџџ*
	elem_type0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1
і
Wtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopV2/EnterEnterLtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1
т
Ntraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/Const_1Const^training/Adam/gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
valueB
 *    *
dtype0*
_output_shapes
: 
ѕ
Otraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgsBroadcastGradientArgsZtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopV2Atraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1
б
Utraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/ConstConst*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ї
Utraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_accStackV2Utraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/Const*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*

stack_name *
_output_shapes
:*
	elem_type0
х
Utraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/EnterEnterUtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
parallel_iterations 

[training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Utraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/Enter?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Shape^training/Adam/gradients/Add*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
_output_shapes
:*
swap_memory(
Ц
Ztraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2`training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1

`training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterUtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
parallel_iterations *
is_constant(
я
@training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/SelectSelectFtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual7training/Adam/gradients/lstm_1/while/mul_2_grad/Reshape?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/zeros*'
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1
ё
Btraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Select_1SelectFtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/zeros7training/Adam/gradients/lstm_1/while/mul_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1
Ш
=training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/SumSum@training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/SelectOtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1
и
Atraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/ReshapeReshape=training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/SumZtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopV2*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Sum_1SumBtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Select_1Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs:1*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
_output_shapes
:*
	keep_dims( *

Tidx0
В
Ctraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Reshape_1Reshape?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Sum_1Atraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Shape_1*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
Tshape0*
_output_shapes
: 
Ю
=training/Adam/gradients/lstm_1/while/clip_by_value_grad/ShapeShape"lstm_1/while/clip_by_value/Minimum*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
out_type0*
_output_shapes
:
Я
?training/Adam/gradients/lstm_1/while/clip_by_value_grad/Shape_1Const^training/Adam/gradients/Sub*-
_class#
!loc:@lstm_1/while/clip_by_value*
valueB *
dtype0*
_output_shapes
: 
х
?training/Adam/gradients/lstm_1/while/clip_by_value_grad/Shape_2Shape7training/Adam/gradients/lstm_1/while/mul_3_grad/Reshape*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
out_type0*
_output_shapes
:
е
Ctraining/Adam/gradients/lstm_1/while/clip_by_value_grad/zeros/ConstConst^training/Adam/gradients/Sub*-
_class#
!loc:@lstm_1/while/clip_by_value*
valueB
 *    *
dtype0*
_output_shapes
: 
О
=training/Adam/gradients/lstm_1/while/clip_by_value_grad/zerosFill?training/Adam/gradients/lstm_1/while/clip_by_value_grad/Shape_2Ctraining/Adam/gradients/lstm_1/while/clip_by_value_grad/zeros/Const*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*

index_type0*'
_output_shapes
:џџџџџџџџџ
д
Dtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqualGreaterEqualOtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopV2Ltraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/Const_1*'
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value
э
Jtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/ConstConst*V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
И
Jtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_accStackV2Jtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/Const*V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*

stack_name *
_output_shapes
:*
	elem_type0
Э
Jtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/EnterEnterJtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
с
Ptraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPushV2StackPushV2Jtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/Enter"lstm_1/while/clip_by_value/Minimum^training/Adam/gradients/Add*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*'
_output_shapes
:џџџџџџџџџ*
swap_memory(
Л
Otraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopV2
StackPopV2Utraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopV2/Enter^training/Adam/gradients/Sub*
	elem_type0*-
_class#
!loc:@lstm_1/while/clip_by_value*'
_output_shapes
:џџџџџџџџџ
№
Utraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopV2/EnterEnterJtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
о
Ltraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/Const_1Const^training/Adam/gradients/Sub*-
_class#
!loc:@lstm_1/while/clip_by_value*
valueB
 *    *
dtype0*
_output_shapes
: 
э
Mtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsXtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopV2?training/Adam/gradients/lstm_1/while/clip_by_value_grad/Shape_1*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Э
Straining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/ConstConst*-
_class#
!loc:@lstm_1/while/clip_by_value*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ё
Straining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_accStackV2Straining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/Const*
_output_shapes
:*
	elem_type0*-
_class#
!loc:@lstm_1/while/clip_by_value*

stack_name 
п
Straining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/EnterEnterStraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
parallel_iterations *
is_constant(

Ytraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPushV2StackPushV2Straining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/Enter=training/Adam/gradients/lstm_1/while/clip_by_value_grad/Shape^training/Adam/gradients/Add*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
_output_shapes
:*
swap_memory(
Р
Xtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopV2
StackPopV2^training/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*-
_class#
!loc:@lstm_1/while/clip_by_value*
_output_shapes
:*
	elem_type0

^training/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopV2/EnterEnterStraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
parallel_iterations 
ч
>training/Adam/gradients/lstm_1/while/clip_by_value_grad/SelectSelectDtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual7training/Adam/gradients/lstm_1/while/mul_3_grad/Reshape=training/Adam/gradients/lstm_1/while/clip_by_value_grad/zeros*-
_class#
!loc:@lstm_1/while/clip_by_value*'
_output_shapes
:џџџџџџџџџ*
T0
щ
@training/Adam/gradients/lstm_1/while/clip_by_value_grad/Select_1SelectDtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual=training/Adam/gradients/lstm_1/while/clip_by_value_grad/zeros7training/Adam/gradients/lstm_1/while/mul_3_grad/Reshape*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*'
_output_shapes
:џџџџџџџџџ
Р
;training/Adam/gradients/lstm_1/while/clip_by_value_grad/SumSum>training/Adam/gradients/lstm_1/while/clip_by_value_grad/SelectMtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs*-
_class#
!loc:@lstm_1/while/clip_by_value*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
а
?training/Adam/gradients/lstm_1/while/clip_by_value_grad/ReshapeReshape;training/Adam/gradients/lstm_1/while/clip_by_value_grad/SumXtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
Tshape0
Ц
=training/Adam/gradients/lstm_1/while/clip_by_value_grad/Sum_1Sum@training/Adam/gradients/lstm_1/while/clip_by_value_grad/Select_1Otraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs:1*-
_class#
!loc:@lstm_1/while/clip_by_value*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Њ
Atraining/Adam/gradients/lstm_1/while/clip_by_value_grad/Reshape_1Reshape=training/Adam/gradients/lstm_1/while/clip_by_value_grad/Sum_1?training/Adam/gradients/lstm_1/while/clip_by_value_grad/Shape_1*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
Tshape0*
_output_shapes
: 

7training/Adam/gradients/lstm_1/while/Tanh_grad/TanhGradTanhGrad>training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPopV29training/Adam/gradients/lstm_1/while/mul_3_grad/Reshape_1*
T0*$
_class
loc:@lstm_1/while/Tanh*'
_output_shapes
:џџџџџџџџџ
В
5training/Adam/gradients/lstm_1/while/add_7_grad/ShapeShapelstm_1/while/BiasAdd_3*
T0*%
_class
loc:@lstm_1/while/add_7*
out_type0*
_output_shapes
:
Г
7training/Adam/gradients/lstm_1/while/add_7_grad/Shape_1Shapelstm_1/while/MatMul_7*
T0*%
_class
loc:@lstm_1/while/add_7*
out_type0*
_output_shapes
:
ш
Etraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2Rtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/add_7*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/add_7*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/add_7*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_7*
parallel_iterations 
с
Qtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/add_7_grad/Shape^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/add_7*
_output_shapes
:*
swap_memory(
Ј
Ptraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_7*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_7*
parallel_iterations 
П
Mtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/Const_1Const*%
_class
loc:@lstm_1/while/add_7*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Mtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/Const_1*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/add_7*

stack_name 
Ы
Mtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc_1*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_7*
parallel_iterations 
ч
Straining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/add_7_grad/Shape_1^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/add_7*
_output_shapes
:*
swap_memory(
Ќ
Rtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_7*
_output_shapes
:*
	elem_type0
ю
Xtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterMtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_7*
parallel_iterations *
is_constant(
Ѓ
3training/Adam/gradients/lstm_1/while/add_7_grad/SumSum9training/Adam/gradients/lstm_1/while/mul_4_grad/Reshape_1Etraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_7*
_output_shapes
:
А
7training/Adam/gradients/lstm_1/while/add_7_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_7_grad/SumPtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/add_7*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ї
5training/Adam/gradients/lstm_1/while/add_7_grad/Sum_1Sum9training/Adam/gradients/lstm_1/while/mul_4_grad/Reshape_1Gtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@lstm_1/while/add_7*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
9training/Adam/gradients/lstm_1/while/add_7_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_7_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/add_7*
Tshape0*'
_output_shapes
:џџџџџџџџџ
в
Gtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/ShapeShapelstm_1/while/add_4*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
out_type0*
_output_shapes
:
у
Itraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_1Const^training/Adam/gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
valueB *
dtype0*
_output_shapes
: 

Itraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_2ShapeAtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Reshape*
_output_shapes
:*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
out_type0
щ
Mtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros/ConstConst^training/Adam/gradients/Sub*
_output_shapes
: *7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
valueB
 *    *
dtype0
ц
Gtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zerosFillItraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_2Mtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros/Const*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*

index_type0*'
_output_shapes
:џџџџџџџџџ*
T0
№
Ktraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual	LessEqualVtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopV2Straining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/Const_1*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*'
_output_shapes
:џџџџџџџџџ
ю
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/ConstConst*P
_classF
Dloc:@lstm_1/while/add_4)loc:@lstm_1/while/clip_by_value_1/Minimum*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Р
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_accStackV2Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/Const*P
_classF
Dloc:@lstm_1/while/add_4)loc:@lstm_1/while/clip_by_value_1/Minimum*

stack_name *
_output_shapes
:*
	elem_type0
х
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/EnterEnterQtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum
щ
Wtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPushV2StackPushV2Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/Enterlstm_1/while/add_4^training/Adam/gradients/Add*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*'
_output_shapes
:џџџџџџџџџ*
swap_memory(
г
Vtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopV2
StackPopV2\training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopV2/Enter^training/Adam/gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*'
_output_shapes
:џџџџџџџџџ*
	elem_type0

\training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopV2/EnterEnterQtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
parallel_iterations *
is_constant(
я
Straining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/Const_1Const^training/Adam/gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Wtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsbtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopV2Itraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum
с
]training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/ConstConst*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
П
]training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_accStackV2]training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/Const*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*

stack_name *
_output_shapes
:*
	elem_type0
§
]training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/EnterEnter]training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
parallel_iterations 
Љ
ctraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPushV2StackPushV2]training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/EnterGtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape^training/Adam/gradients/Add*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
_output_shapes
:*
swap_memory(
о
btraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopV2
StackPopV2htraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum
 
htraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopV2/EnterEnter]training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
parallel_iterations *
is_constant(

Htraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SelectSelectKtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqualAtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/ReshapeGtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*'
_output_shapes
:џџџџџџџџџ

Jtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Select_1SelectKtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqualGtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zerosAtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Reshape*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*'
_output_shapes
:џџџџџџџџџ
ш
Etraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SumSumHtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SelectWtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum
ј
Itraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/ReshapeReshapeEtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Sumbtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopV2*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
ю
Gtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Sum_1SumJtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Select_1Ytraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs:1*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
_output_shapes
:*
	keep_dims( *

Tidx0
в
Ktraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Reshape_1ReshapeGtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Sum_1Itraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_1*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
Tshape0*
_output_shapes
: 
щ
Btraining/Adam/gradients/lstm_1/while/Switch_4_grad_1/NextIterationNextIteration9training/Adam/gradients/lstm_1/while/mul_2_grad/Reshape_1*
T0*'
_class
loc:@lstm_1/while/Merge_4*'
_output_shapes
:џџџџџџџџџ
Ю
Etraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/ShapeShapelstm_1/while/add_2*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
out_type0*
_output_shapes
:
п
Gtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_1Const^training/Adam/gradients/Sub*
dtype0*
_output_shapes
: *5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
valueB 
§
Gtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_2Shape?training/Adam/gradients/lstm_1/while/clip_by_value_grad/Reshape*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
out_type0*
_output_shapes
:*
T0
х
Ktraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros/ConstConst^training/Adam/gradients/Sub*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
valueB
 *    *
dtype0*
_output_shapes
: 
о
Etraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/zerosFillGtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_2Ktraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros/Const*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*

index_type0*'
_output_shapes
:џџџџџџџџџ
ш
Itraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual	LessEqualTtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopV2Qtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/Const_1*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ
ъ
Otraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/ConstConst*N
_classD
Bloc:@lstm_1/while/add_2'loc:@lstm_1/while/clip_by_value/Minimum*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
К
Otraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_accStackV2Otraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/Const*N
_classD
Bloc:@lstm_1/while/add_2'loc:@lstm_1/while/clip_by_value/Minimum*

stack_name *
_output_shapes
:*
	elem_type0
п
Otraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/EnterEnterOtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
parallel_iterations 
у
Utraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPushV2StackPushV2Otraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/Enterlstm_1/while/add_2^training/Adam/gradients/Add*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ*
swap_memory(
Э
Ttraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopV2
StackPopV2Ztraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopV2/Enter^training/Adam/gradients/Sub*'
_output_shapes
:џџџџџџџџџ*
	elem_type0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum

Ztraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopV2/EnterEnterOtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
parallel_iterations 
ы
Qtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/Const_1Const^training/Adam/gradients/Sub*
_output_shapes
: *5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
valueB
 *  ?*
dtype0

Utraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs`training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopV2Gtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_1*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
н
[training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/ConstConst*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Й
[training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_accStackV2[training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/Const*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*

stack_name *
_output_shapes
:*
	elem_type0
ї
[training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/EnterEnter[training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
parallel_iterations *
is_constant(
Ё
atraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPushV2StackPushV2[training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/EnterEtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape^training/Adam/gradients/Add*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
_output_shapes
:*
swap_memory(*
T0
и
`training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ftraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
_output_shapes
:*
	elem_type0

ftraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopV2/EnterEnter[training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_acc*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context

Ftraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/SelectSelectItraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual?training/Adam/gradients/lstm_1/while/clip_by_value_grad/ReshapeEtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ*
T0

Htraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Select_1SelectItraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqualEtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros?training/Adam/gradients/lstm_1/while/clip_by_value_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum
р
Ctraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/SumSumFtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/SelectUtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
_output_shapes
:*
	keep_dims( *

Tidx0
№
Gtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/ReshapeReshapeCtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Sum`training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
Tshape0
ц
Etraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Sum_1SumHtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Select_1Wtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
_output_shapes
:*
	keep_dims( *

Tidx0
Ъ
Itraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Reshape_1ReshapeEtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Sum_1Gtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_1*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
Tshape0*
_output_shapes
: 
В
5training/Adam/gradients/lstm_1/while/add_5_grad/ShapeShapelstm_1/while/BiasAdd_2*
T0*%
_class
loc:@lstm_1/while/add_5*
out_type0*
_output_shapes
:
Г
7training/Adam/gradients/lstm_1/while/add_5_grad/Shape_1Shapelstm_1/while/MatMul_6*
_output_shapes
:*
T0*%
_class
loc:@lstm_1/while/add_5*
out_type0
ш
Etraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2Rtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/add_5
Н
Ktraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/add_5*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/add_5*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc*
T0*%
_class
loc:@lstm_1/while/add_5*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
с
Qtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/add_5_grad/Shape^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/add_5*
_output_shapes
:*
swap_memory(
Ј
Ptraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
	elem_type0*%
_class
loc:@lstm_1/while/add_5*
_output_shapes
:
ъ
Vtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_5*
parallel_iterations 
П
Mtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/Const_1Const*%
_class
loc:@lstm_1/while/add_5*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Mtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/Const_1*%
_class
loc:@lstm_1/while/add_5*

stack_name *
_output_shapes
:*
	elem_type0
Ы
Mtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1*
T0*%
_class
loc:@lstm_1/while/add_5*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
ч
Straining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/add_5_grad/Shape_1^training/Adam/gradients/Add*
_output_shapes
:*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/add_5
Ќ
Rtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/add_5
ю
Xtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterMtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1*
T0*%
_class
loc:@lstm_1/while/add_5*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
Ё
3training/Adam/gradients/lstm_1/while/add_5_grad/SumSum7training/Adam/gradients/lstm_1/while/Tanh_grad/TanhGradEtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_5*
_output_shapes
:
А
7training/Adam/gradients/lstm_1/while/add_5_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_5_grad/SumPtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/add_5*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ѕ
5training/Adam/gradients/lstm_1/while/add_5_grad/Sum_1Sum7training/Adam/gradients/lstm_1/while/Tanh_grad/TanhGradGtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_5*
_output_shapes
:
Ж
9training/Adam/gradients/lstm_1/while/add_5_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_5_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/add_5*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ю
?training/Adam/gradients/lstm_1/while/BiasAdd_3_grad/BiasAddGradBiasAddGrad7training/Adam/gradients/lstm_1/while/add_7_grad/Reshape*
T0*)
_class
loc:@lstm_1/while/BiasAdd_3*
data_formatNHWC*
_output_shapes
:
С
9training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMulMatMul9training/Adam/gradients/lstm_1/while/add_7_grad/Reshape_1?training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul/Enter*
T0*(
_class
loc:@lstm_1/while/MatMul_7*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul/EnterEnterlstm_1/strided_slice_7*
T0*(
_class
loc:@lstm_1/while/MatMul_7*
parallel_iterations *
is_constant(*
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
С
;training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV29training/Adam/gradients/lstm_1/while/add_7_grad/Reshape_1*
transpose_b( *
T0*(
_class
loc:@lstm_1/while/MatMul_7*
_output_shapes

:*
transpose_a(
д
Atraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/ConstConst*
_output_shapes
: *F
_class<
:loc:@lstm_1/while/Identity_3loc:@lstm_1/while/MatMul_7*
valueB :
џџџџџџџџџ*
dtype0

Atraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/f_accStackV2Atraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/Const*
_output_shapes
:*
	elem_type0*F
_class<
:loc:@lstm_1/while/Identity_3loc:@lstm_1/while/MatMul_7*

stack_name 
Ж
Atraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/EnterEnterAtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/f_acc*
T0*(
_class
loc:@lstm_1/while/MatMul_7*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
П
Gtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPushV2StackPushV2Atraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/Enterlstm_1/while/Identity_3^training/Adam/gradients/Add*
T0*(
_class
loc:@lstm_1/while/MatMul_7*'
_output_shapes
:џџџџџџџџџ*
swap_memory(
Є
Ftraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV2
StackPopV2Ltraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV2/Enter^training/Adam/gradients/Sub*(
_class
loc:@lstm_1/while/MatMul_7*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
й
Ltraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV2/EnterEnterAtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/f_acc*
T0*(
_class
loc:@lstm_1/while/MatMul_7*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
Ў
5training/Adam/gradients/lstm_1/while/add_4_grad/ShapeShapelstm_1/while/mul_1*
T0*%
_class
loc:@lstm_1/while/add_4*
out_type0*
_output_shapes
:
П
7training/Adam/gradients/lstm_1/while/add_4_grad/Shape_1Const^training/Adam/gradients/Sub*
dtype0*
_output_shapes
: *%
_class
loc:@lstm_1/while/add_4*
valueB 
Э
Etraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopV27training/Adam/gradients/lstm_1/while/add_4_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/add_4
Н
Ktraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/add_4*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/add_4*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc*
T0*%
_class
loc:@lstm_1/while/add_4*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
с
Qtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/add_4_grad/Shape^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/add_4*
_output_shapes
:*
swap_memory(
Ј
Ptraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_4*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_4*
parallel_iterations 
Г
3training/Adam/gradients/lstm_1/while/add_4_grad/SumSumItraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/ReshapeEtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_4
А
7training/Adam/gradients/lstm_1/while/add_4_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_4_grad/SumPtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/add_4*
Tshape0
З
5training/Adam/gradients/lstm_1/while/add_4_grad/Sum_1SumItraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/ReshapeGtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_4

9training/Adam/gradients/lstm_1/while/add_4_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_4_grad/Sum_17training/Adam/gradients/lstm_1/while/add_4_grad/Shape_1*
T0*%
_class
loc:@lstm_1/while/add_4*
Tshape0*
_output_shapes
: 
Ќ
5training/Adam/gradients/lstm_1/while/add_2_grad/ShapeShapelstm_1/while/mul*
T0*%
_class
loc:@lstm_1/while/add_2*
out_type0*
_output_shapes
:
П
7training/Adam/gradients/lstm_1/while/add_2_grad/Shape_1Const^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_2*
valueB *
dtype0*
_output_shapes
: 
Э
Etraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopV27training/Adam/gradients/lstm_1/while/add_2_grad/Shape_1*
T0*%
_class
loc:@lstm_1/while/add_2*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/add_2*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/add_2*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_2*
parallel_iterations 
с
Qtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/add_2_grad/Shape^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/add_2*
_output_shapes
:*
swap_memory(
Ј
Ptraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
	elem_type0*%
_class
loc:@lstm_1/while/add_2*
_output_shapes
:
ъ
Vtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc*
T0*%
_class
loc:@lstm_1/while/add_2*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
Б
3training/Adam/gradients/lstm_1/while/add_2_grad/SumSumGtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/ReshapeEtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_2
А
7training/Adam/gradients/lstm_1/while/add_2_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_2_grad/SumPtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/add_2*
Tshape0
Е
5training/Adam/gradients/lstm_1/while/add_2_grad/Sum_1SumGtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/ReshapeGtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_2*
_output_shapes
:

9training/Adam/gradients/lstm_1/while/add_2_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_2_grad/Sum_17training/Adam/gradients/lstm_1/while/add_2_grad/Shape_1*
T0*%
_class
loc:@lstm_1/while/add_2*
Tshape0*
_output_shapes
: 
ю
?training/Adam/gradients/lstm_1/while/BiasAdd_2_grad/BiasAddGradBiasAddGrad7training/Adam/gradients/lstm_1/while/add_5_grad/Reshape*
T0*)
_class
loc:@lstm_1/while/BiasAdd_2*
data_formatNHWC*
_output_shapes
:
С
9training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMulMatMul9training/Adam/gradients/lstm_1/while/add_5_grad/Reshape_1?training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMul/Enter*
T0*(
_class
loc:@lstm_1/while/MatMul_6*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMul/EnterEnterlstm_1/strided_slice_6*
T0*(
_class
loc:@lstm_1/while/MatMul_6*
parallel_iterations *
is_constant(*
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
С
;training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV29training/Adam/gradients/lstm_1/while/add_5_grad/Reshape_1*
T0*(
_class
loc:@lstm_1/while/MatMul_6*
_output_shapes

:*
transpose_a(*
transpose_b( 
П
9training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMulMatMul7training/Adam/gradients/lstm_1/while/add_7_grad/Reshape?training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul/Enter*
T0*(
_class
loc:@lstm_1/while/MatMul_3*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b(
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul/EnterEnterlstm_1/strided_slice_3*
is_constant(*
_output_shapes

:@*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_3*
parallel_iterations 
П
;training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV27training/Adam/gradients/lstm_1/while/add_7_grad/Reshape*
_output_shapes

:@*
transpose_a(*
transpose_b( *
T0*(
_class
loc:@lstm_1/while/MatMul_3
л
Atraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/ConstConst*M
_classC
Aloc:@lstm_1/while/MatMul_3#loc:@lstm_1/while/TensorArrayReadV3*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Atraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_accStackV2Atraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*M
_classC
Aloc:@lstm_1/while/MatMul_3#loc:@lstm_1/while/TensorArrayReadV3
Ж
Atraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/EnterEnterAtraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_3*
parallel_iterations 
Ц
Gtraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPushV2StackPushV2Atraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/Enterlstm_1/while/TensorArrayReadV3^training/Adam/gradients/Add*'
_output_shapes
:џџџџџџџџџ@*
swap_memory(*
T0*(
_class
loc:@lstm_1/while/MatMul_3
Є
Ftraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV2
StackPopV2Ltraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV2/Enter^training/Adam/gradients/Sub*
	elem_type0*(
_class
loc:@lstm_1/while/MatMul_3*'
_output_shapes
:џџџџџџџџџ@
й
Ltraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV2/EnterEnterAtraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_acc*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_3*
parallel_iterations *
is_constant(
Н
?training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_accConst*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter*
valueB*    *
dtype0
г
Atraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_1Enter?training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter*
parallel_iterations *
is_constant( *
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0
З
Atraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_2MergeAtraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_1Gtraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/NextIteration*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter*
N*
_output_shapes

:: 

@training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/SwitchSwitchAtraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter* 
_output_shapes
::

=training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/AddAddBtraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/Switch:1?training/Adam/gradients/lstm_1/while/BiasAdd_3_grad/BiasAddGrad*
_output_shapes
:*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter
э
Gtraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/NextIterationNextIteration=training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/Add*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter*
_output_shapes
:*
T0
с
Atraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_3Exit@training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/Switch*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter*
_output_shapes
:*
T0
У
>training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_accConst*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*
valueB*    *
dtype0*
_output_shapes

:
д
@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc*
T0*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*
parallel_iterations *
is_constant( *
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
З
@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/NextIteration*
T0*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*
N* 
_output_shapes
:: 

?training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*(
_output_shapes
::

<training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1*
_output_shapes

:*
T0*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/Add*
T0*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*
_output_shapes

:
т
@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/Switch*
T0*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*
_output_shapes

:
Н
5training/Adam/gradients/lstm_1/while/mul_1_grad/ShapeConst^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_1*
valueB *
dtype0*
_output_shapes
: 
А
7training/Adam/gradients/lstm_1/while/mul_1_grad/Shape_1Shapelstm_1/while/add_3*
T0*%
_class
loc:@lstm_1/while/mul_1*
out_type0*
_output_shapes
:
Ы
Etraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/lstm_1/while/mul_1_grad/ShapePtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/mul_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/mul_1*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_1*
parallel_iterations *
is_constant(
у
Qtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/Enter7training/Adam/gradients/lstm_1/while/mul_1_grad/Shape_1^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_1*
_output_shapes
:*
swap_memory(
Ј
Ptraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_1*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_1*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context

3training/Adam/gradients/lstm_1/while/mul_1_grad/MulMul7training/Adam/gradients/lstm_1/while/add_4_grad/Reshape>training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_1*'
_output_shapes
:џџџџџџџџџ
Ф
9training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/ConstConst*>
_class4
2loc:@lstm_1/while/add_3loc:@lstm_1/while/mul_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
ў
9training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/f_accStackV29training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/Const*>
_class4
2loc:@lstm_1/while/add_3loc:@lstm_1/while/mul_1*

stack_name *
_output_shapes
:*
	elem_type0
Ѓ
9training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/EnterEnter9training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_1*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
Ї
?training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPushV2StackPushV29training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/Enterlstm_1/while/add_3^training/Adam/gradients/Add*'
_output_shapes
:џџџџџџџџџ*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/mul_1

>training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPopV2
StackPopV2Dtraining/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_1*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
Ц
Dtraining/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPopV2/EnterEnter9training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_1

3training/Adam/gradients/lstm_1/while/mul_1_grad/SumSum3training/Adam/gradients/lstm_1/while/mul_1_grad/MulEtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs*
T0*%
_class
loc:@lstm_1/while/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0

7training/Adam/gradients/lstm_1/while/mul_1_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/mul_1_grad/Sum5training/Adam/gradients/lstm_1/while/mul_1_grad/Shape*
T0*%
_class
loc:@lstm_1/while/mul_1*
Tshape0*
_output_shapes
: 

5training/Adam/gradients/lstm_1/while/mul_1_grad/Mul_1Mul;training/Adam/gradients/lstm_1/while/mul_1_grad/Mul_1/Const7training/Adam/gradients/lstm_1/while/add_4_grad/Reshape*
T0*%
_class
loc:@lstm_1/while/mul_1*'
_output_shapes
:џџџџџџџџџ
Х
;training/Adam/gradients/lstm_1/while/mul_1_grad/Mul_1/ConstConst^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_1*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
Ѓ
5training/Adam/gradients/lstm_1/while/mul_1_grad/Sum_1Sum5training/Adam/gradients/lstm_1/while/mul_1_grad/Mul_1Gtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/mul_1
Д
9training/Adam/gradients/lstm_1/while/mul_1_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/mul_1_grad/Sum_1Ptraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/mul_1*
Tshape0
Й
3training/Adam/gradients/lstm_1/while/mul_grad/ShapeConst^training/Adam/gradients/Sub*
dtype0*
_output_shapes
: *#
_class
loc:@lstm_1/while/mul*
valueB 
Ќ
5training/Adam/gradients/lstm_1/while/mul_grad/Shape_1Shapelstm_1/while/add_1*
_output_shapes
:*
T0*#
_class
loc:@lstm_1/while/mul*
out_type0
У
Ctraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgsBroadcastGradientArgs3training/Adam/gradients/lstm_1/while/mul_grad/ShapeNtraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*#
_class
loc:@lstm_1/while/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Й
Itraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/ConstConst*#
_class
loc:@lstm_1/while/mul*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Itraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_accStackV2Itraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/Const*#
_class
loc:@lstm_1/while/mul*

stack_name *
_output_shapes
:*
	elem_type0
С
Itraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/EnterEnterItraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*#
_class
loc:@lstm_1/while/mul*
parallel_iterations 
л
Otraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Itraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/mul_grad/Shape_1^training/Adam/gradients/Add*
T0*#
_class
loc:@lstm_1/while/mul*
_output_shapes
:*
swap_memory(
Ђ
Ntraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ttraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*#
_class
loc:@lstm_1/while/mul*
_output_shapes
:*
	elem_type0
ф
Ttraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterItraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*#
_class
loc:@lstm_1/while/mul*
parallel_iterations 

1training/Adam/gradients/lstm_1/while/mul_grad/MulMul7training/Adam/gradients/lstm_1/while/add_2_grad/Reshape<training/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPopV2*
T0*#
_class
loc:@lstm_1/while/mul*'
_output_shapes
:џџџџџџџџџ
Р
7training/Adam/gradients/lstm_1/while/mul_grad/Mul/ConstConst*<
_class2
0loc:@lstm_1/while/add_1loc:@lstm_1/while/mul*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
ј
7training/Adam/gradients/lstm_1/while/mul_grad/Mul/f_accStackV27training/Adam/gradients/lstm_1/while/mul_grad/Mul/Const*<
_class2
0loc:@lstm_1/while/add_1loc:@lstm_1/while/mul*

stack_name *
_output_shapes
:*
	elem_type0

7training/Adam/gradients/lstm_1/while/mul_grad/Mul/EnterEnter7training/Adam/gradients/lstm_1/while/mul_grad/Mul/f_acc*
T0*#
_class
loc:@lstm_1/while/mul*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
Ё
=training/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPushV2StackPushV27training/Adam/gradients/lstm_1/while/mul_grad/Mul/Enterlstm_1/while/add_1^training/Adam/gradients/Add*#
_class
loc:@lstm_1/while/mul*'
_output_shapes
:џџџџџџџџџ*
swap_memory(*
T0

<training/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPopV2
StackPopV2Btraining/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*#
_class
loc:@lstm_1/while/mul*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
Р
Btraining/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPopV2/EnterEnter7training/Adam/gradients/lstm_1/while/mul_grad/Mul/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*#
_class
loc:@lstm_1/while/mul*
parallel_iterations 

1training/Adam/gradients/lstm_1/while/mul_grad/SumSum1training/Adam/gradients/lstm_1/while/mul_grad/MulCtraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs*
T0*#
_class
loc:@lstm_1/while/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
ќ
5training/Adam/gradients/lstm_1/while/mul_grad/ReshapeReshape1training/Adam/gradients/lstm_1/while/mul_grad/Sum3training/Adam/gradients/lstm_1/while/mul_grad/Shape*
T0*#
_class
loc:@lstm_1/while/mul*
Tshape0*
_output_shapes
: 

3training/Adam/gradients/lstm_1/while/mul_grad/Mul_1Mul9training/Adam/gradients/lstm_1/while/mul_grad/Mul_1/Const7training/Adam/gradients/lstm_1/while/add_2_grad/Reshape*
T0*#
_class
loc:@lstm_1/while/mul*'
_output_shapes
:џџџџџџџџџ
С
9training/Adam/gradients/lstm_1/while/mul_grad/Mul_1/ConstConst^training/Adam/gradients/Sub*
dtype0*
_output_shapes
: *#
_class
loc:@lstm_1/while/mul*
valueB
 *ЭЬL>

3training/Adam/gradients/lstm_1/while/mul_grad/Sum_1Sum3training/Adam/gradients/lstm_1/while/mul_grad/Mul_1Etraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*#
_class
loc:@lstm_1/while/mul
Ќ
7training/Adam/gradients/lstm_1/while/mul_grad/Reshape_1Reshape3training/Adam/gradients/lstm_1/while/mul_grad/Sum_1Ntraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*#
_class
loc:@lstm_1/while/mul*
Tshape0
П
9training/Adam/gradients/lstm_1/while/MatMul_2_grad/MatMulMatMul7training/Adam/gradients/lstm_1/while/add_5_grad/Reshape?training/Adam/gradients/lstm_1/while/MatMul_2_grad/MatMul/Enter*
transpose_b(*
T0*(
_class
loc:@lstm_1/while/MatMul_2*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( 
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_2_grad/MatMul/EnterEnterlstm_1/strided_slice_2*
is_constant(*
_output_shapes

:@*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_2*
parallel_iterations 
П
;training/Adam/gradients/lstm_1/while/MatMul_2_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV27training/Adam/gradients/lstm_1/while/add_5_grad/Reshape*
T0*(
_class
loc:@lstm_1/while/MatMul_2*
_output_shapes

:@*
transpose_a(*
transpose_b( 
Н
?training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_accConst*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter*
valueB*    *
dtype0*
_output_shapes
:
г
Atraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc_1Enter?training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc*
is_constant( *
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter*
parallel_iterations 
З
Atraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc_2MergeAtraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc_1Gtraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/NextIteration*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter*
N*
_output_shapes

:: 

@training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/SwitchSwitchAtraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter* 
_output_shapes
::

=training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/AddAddBtraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/Switch:1?training/Adam/gradients/lstm_1/while/BiasAdd_2_grad/BiasAddGrad*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter*
_output_shapes
:
э
Gtraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/NextIterationNextIteration=training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/Add*
_output_shapes
:*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter
с
Atraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc_3Exit@training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/Switch*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter*
_output_shapes
:
У
>training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_accConst*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*
valueB*    *
dtype0*
_output_shapes

:
д
@training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc*
is_constant( *
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*
parallel_iterations 
З
@training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/NextIteration*
T0*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*
N* 
_output_shapes
:: 

?training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*(
_output_shapes
::

<training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMul_1*
T0*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*
_output_shapes

:
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/Add*
T0*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*
_output_shapes

:
т
@training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/Switch*
T0*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*
_output_shapes

:
У
>training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_accConst*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
valueB@*    *
dtype0*
_output_shapes

:@
д
@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
parallel_iterations *
is_constant( *
_output_shapes

:@*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0
З
@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/NextIteration*
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
N* 
_output_shapes
:@: 

?training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*(
_output_shapes
:@:@

<training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1*
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
_output_shapes

:@
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/Add*
_output_shapes

:@*
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter
т
@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/Switch*
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
_output_shapes

:@
А
:training/Adam/gradients/lstm_1/strided_slice_11_grad/ShapeConst*
dtype0*
_output_shapes
:**
_class 
loc:@lstm_1/strided_slice_11*
valueB:
ћ
Etraining/Adam/gradients/lstm_1/strided_slice_11_grad/StridedSliceGradStridedSliceGrad:training/Adam/gradients/lstm_1/strided_slice_11_grad/Shapelstm_1/strided_slice_11/stacklstm_1/strided_slice_11/stack_1lstm_1/strided_slice_11/stack_2Atraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_3*
Index0*
T0**
_class 
loc:@lstm_1/strided_slice_11*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:
Е
9training/Adam/gradients/lstm_1/strided_slice_7_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_7*
valueB"      *
dtype0*
_output_shapes
:
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_7_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_7_grad/Shapelstm_1/strided_slice_7/stacklstm_1/strided_slice_7/stack_1lstm_1/strided_slice_7/stack_2@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_3*
_output_shapes

:*
Index0*
T0*)
_class
loc:@lstm_1/strided_slice_7*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
В
5training/Adam/gradients/lstm_1/while/add_3_grad/ShapeShapelstm_1/while/BiasAdd_1*
T0*%
_class
loc:@lstm_1/while/add_3*
out_type0*
_output_shapes
:
Г
7training/Adam/gradients/lstm_1/while/add_3_grad/Shape_1Shapelstm_1/while/MatMul_5*
T0*%
_class
loc:@lstm_1/while/add_3*
out_type0*
_output_shapes
:
ш
Etraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2Rtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/add_3*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/add_3*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/Const*
	elem_type0*%
_class
loc:@lstm_1/while/add_3*

stack_name *
_output_shapes
:
Ч
Ktraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_3
с
Qtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/add_3_grad/Shape^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/add_3*
_output_shapes
:*
swap_memory(
Ј
Ptraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_3*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc*%
_class
loc:@lstm_1/while/add_3*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0
П
Mtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/Const_1Const*
_output_shapes
: *%
_class
loc:@lstm_1/while/add_3*
valueB :
џџџџџџџџџ*
dtype0

Mtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/Const_1*%
_class
loc:@lstm_1/while/add_3*

stack_name *
_output_shapes
:*
	elem_type0
Ы
Mtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc_1*
T0*%
_class
loc:@lstm_1/while/add_3*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
ч
Straining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/add_3_grad/Shape_1^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/add_3*
_output_shapes
:*
swap_memory(
Ќ
Rtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*
	elem_type0*%
_class
loc:@lstm_1/while/add_3*
_output_shapes
:
ю
Xtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterMtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc_1*%
_class
loc:@lstm_1/while/add_3*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0
Ѓ
3training/Adam/gradients/lstm_1/while/add_3_grad/SumSum9training/Adam/gradients/lstm_1/while/mul_1_grad/Reshape_1Etraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_3*
_output_shapes
:
А
7training/Adam/gradients/lstm_1/while/add_3_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_3_grad/SumPtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/add_3*
Tshape0
Ї
5training/Adam/gradients/lstm_1/while/add_3_grad/Sum_1Sum9training/Adam/gradients/lstm_1/while/mul_1_grad/Reshape_1Gtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/add_3*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ж
9training/Adam/gradients/lstm_1/while/add_3_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_3_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2_1*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/add_3*
Tshape0
А
5training/Adam/gradients/lstm_1/while/add_1_grad/ShapeShapelstm_1/while/BiasAdd*
_output_shapes
:*
T0*%
_class
loc:@lstm_1/while/add_1*
out_type0
Г
7training/Adam/gradients/lstm_1/while/add_1_grad/Shape_1Shapelstm_1/while/MatMul_4*
_output_shapes
:*
T0*%
_class
loc:@lstm_1/while/add_1*
out_type0
ш
Etraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2Rtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/add_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/add_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/add_1*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_1*
parallel_iterations 
с
Qtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/add_1_grad/Shape^training/Adam/gradients/Add*
_output_shapes
:*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/add_1
Ј
Ptraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_1*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_1*
parallel_iterations 
П
Mtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Const_1Const*%
_class
loc:@lstm_1/while/add_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Mtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Const_1*

stack_name *
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/add_1
Ы
Mtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*%
_class
loc:@lstm_1/while/add_1*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
ч
Straining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/add_1_grad/Shape_1^training/Adam/gradients/Add*%
_class
loc:@lstm_1/while/add_1*
_output_shapes
:*
swap_memory(*
T0
Ќ
Rtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_1*
_output_shapes
:*
	elem_type0
ю
Xtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterMtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_1
Ё
3training/Adam/gradients/lstm_1/while/add_1_grad/SumSum7training/Adam/gradients/lstm_1/while/mul_grad/Reshape_1Etraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs*
T0*%
_class
loc:@lstm_1/while/add_1*
_output_shapes
:*
	keep_dims( *

Tidx0
А
7training/Adam/gradients/lstm_1/while/add_1_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_1_grad/SumPtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/add_1*
Tshape0
Ѕ
5training/Adam/gradients/lstm_1/while/add_1_grad/Sum_1Sum7training/Adam/gradients/lstm_1/while/mul_grad/Reshape_1Gtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@lstm_1/while/add_1*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
9training/Adam/gradients/lstm_1/while/add_1_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_1_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/add_1*
Tshape0*'
_output_shapes
:џџџџџџџџџ
У
>training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_accConst*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
valueB@*    *
dtype0*
_output_shapes

:@
д
@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc*
parallel_iterations *
is_constant( *
_output_shapes

:@*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter
З
@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/NextIteration* 
_output_shapes
:@: *
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
N

?training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*(
_output_shapes
:@:@*
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter

<training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_2_grad/MatMul_1*
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
_output_shapes

:@
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/Add*
_output_shapes

:@*
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter
т
@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
_output_shapes

:@*
T0
А
:training/Adam/gradients/lstm_1/strided_slice_10_grad/ShapeConst**
_class 
loc:@lstm_1/strided_slice_10*
valueB:*
dtype0*
_output_shapes
:
ћ
Etraining/Adam/gradients/lstm_1/strided_slice_10_grad/StridedSliceGradStridedSliceGrad:training/Adam/gradients/lstm_1/strided_slice_10_grad/Shapelstm_1/strided_slice_10/stacklstm_1/strided_slice_10/stack_1lstm_1/strided_slice_10/stack_2Atraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc_3*
Index0*
T0**
_class 
loc:@lstm_1/strided_slice_10*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:
Е
9training/Adam/gradients/lstm_1/strided_slice_6_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_6*
valueB"      *
dtype0*
_output_shapes
:
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_6_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_6_grad/Shapelstm_1/strided_slice_6/stacklstm_1/strided_slice_6/stack_1lstm_1/strided_slice_6/stack_2@training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc_3*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:*
T0*
Index0*)
_class
loc:@lstm_1/strided_slice_6
Е
9training/Adam/gradients/lstm_1/strided_slice_3_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_3*
valueB"@      *
dtype0*
_output_shapes
:
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_3_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_3_grad/Shapelstm_1/strided_slice_3/stacklstm_1/strided_slice_3/stack_1lstm_1/strided_slice_3/stack_2@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_3*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:@*
T0*
Index0*)
_class
loc:@lstm_1/strided_slice_3
ю
?training/Adam/gradients/lstm_1/while/BiasAdd_1_grad/BiasAddGradBiasAddGrad7training/Adam/gradients/lstm_1/while/add_3_grad/Reshape*
_output_shapes
:*
T0*)
_class
loc:@lstm_1/while/BiasAdd_1*
data_formatNHWC
С
9training/Adam/gradients/lstm_1/while/MatMul_5_grad/MatMulMatMul9training/Adam/gradients/lstm_1/while/add_3_grad/Reshape_1?training/Adam/gradients/lstm_1/while/MatMul_5_grad/MatMul/Enter*
T0*(
_class
loc:@lstm_1/while/MatMul_5*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_5_grad/MatMul/EnterEnterlstm_1/strided_slice_5*
is_constant(*
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_5*
parallel_iterations 
С
;training/Adam/gradients/lstm_1/while/MatMul_5_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV29training/Adam/gradients/lstm_1/while/add_3_grad/Reshape_1*
T0*(
_class
loc:@lstm_1/while/MatMul_5*
_output_shapes

:*
transpose_a(*
transpose_b( 
ъ
=training/Adam/gradients/lstm_1/while/BiasAdd_grad/BiasAddGradBiasAddGrad7training/Adam/gradients/lstm_1/while/add_1_grad/Reshape*
data_formatNHWC*
_output_shapes
:*
T0*'
_class
loc:@lstm_1/while/BiasAdd
С
9training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMulMatMul9training/Adam/gradients/lstm_1/while/add_1_grad/Reshape_1?training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMul/Enter*(
_class
loc:@lstm_1/while/MatMul_4*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMul/EnterEnterlstm_1/strided_slice_4*
is_constant(*
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_4*
parallel_iterations 
С
;training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV29training/Adam/gradients/lstm_1/while/add_1_grad/Reshape_1*
T0*(
_class
loc:@lstm_1/while/MatMul_4*
_output_shapes

:*
transpose_a(*
transpose_b( 
Е
9training/Adam/gradients/lstm_1/strided_slice_2_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_2*
valueB"@      *
dtype0*
_output_shapes
:
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_2_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_2_grad/Shapelstm_1/strided_slice_2/stacklstm_1/strided_slice_2/stack_1lstm_1/strided_slice_2/stack_2@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_3*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:@*
T0*
Index0*)
_class
loc:@lstm_1/strided_slice_2
П
9training/Adam/gradients/lstm_1/while/MatMul_1_grad/MatMulMatMul7training/Adam/gradients/lstm_1/while/add_3_grad/Reshape?training/Adam/gradients/lstm_1/while/MatMul_1_grad/MatMul/Enter*
T0*(
_class
loc:@lstm_1/while/MatMul_1*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b(
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_1_grad/MatMul/EnterEnterlstm_1/strided_slice_1*
is_constant(*
_output_shapes

:@*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_1*
parallel_iterations 
П
;training/Adam/gradients/lstm_1/while/MatMul_1_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV27training/Adam/gradients/lstm_1/while/add_3_grad/Reshape*
_output_shapes

:@*
transpose_a(*
transpose_b( *
T0*(
_class
loc:@lstm_1/while/MatMul_1
Н
?training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_accConst*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter*
valueB*    *
dtype0*
_output_shapes
:
г
Atraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_1Enter?training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter*
parallel_iterations *
is_constant( *
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
З
Atraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_2MergeAtraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_1Gtraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/NextIteration*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter*
N*
_output_shapes

:: 

@training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/SwitchSwitchAtraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter* 
_output_shapes
::

=training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/AddAddBtraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/Switch:1?training/Adam/gradients/lstm_1/while/BiasAdd_1_grad/BiasAddGrad*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter*
_output_shapes
:
э
Gtraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/NextIterationNextIteration=training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/Add*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter*
_output_shapes
:
с
Atraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_3Exit@training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/Switch*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter*
_output_shapes
:
У
>training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_accConst*
dtype0*
_output_shapes

:*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter*
valueB*    
д
@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc*
is_constant( *
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter*
parallel_iterations 
З
@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/NextIteration*
T0*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter*
N* 
_output_shapes
:: 

?training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*(
_output_shapes
::*
T0*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter

<training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_5_grad/MatMul_1*
_output_shapes

:*
T0*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/Add*
_output_shapes

:*
T0*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter
т
@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter*
_output_shapes

:*
T0
Й
7training/Adam/gradients/lstm_1/while/MatMul_grad/MatMulMatMul7training/Adam/gradients/lstm_1/while/add_1_grad/Reshape=training/Adam/gradients/lstm_1/while/MatMul_grad/MatMul/Enter*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b(*
T0*&
_class
loc:@lstm_1/while/MatMul

=training/Adam/gradients/lstm_1/while/MatMul_grad/MatMul/EnterEnterlstm_1/strided_slice*
T0*&
_class
loc:@lstm_1/while/MatMul*
parallel_iterations *
is_constant(*
_output_shapes

:@*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
Л
9training/Adam/gradients/lstm_1/while/MatMul_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV27training/Adam/gradients/lstm_1/while/add_1_grad/Reshape*&
_class
loc:@lstm_1/while/MatMul*
_output_shapes

:@*
transpose_a(*
transpose_b( *
T0
Й
=training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_accConst*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter*
valueB*    *
dtype0*
_output_shapes
:
Э
?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_1Enter=training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc*
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter*
parallel_iterations *
is_constant( *
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
Џ
?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_2Merge?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_1Etraining/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/NextIteration*
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter*
N*
_output_shapes

:: 

>training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/SwitchSwitch?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter* 
_output_shapes
::

;training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/AddAdd@training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/Switch:1=training/Adam/gradients/lstm_1/while/BiasAdd_grad/BiasAddGrad*
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter*
_output_shapes
:
ч
Etraining/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/NextIterationNextIteration;training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/Add*
_output_shapes
:*
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter
л
?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_3Exit>training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/Switch*
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter*
_output_shapes
:
ї
training/Adam/gradients/AddN_7AddN9training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul9training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMul9training/Adam/gradients/lstm_1/while/MatMul_5_grad/MatMul9training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMul*
T0*(
_class
loc:@lstm_1/while/MatMul_7*
N*'
_output_shapes
:џџџџџџџџџ
У
>training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_accConst*
dtype0*
_output_shapes

:*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter*
valueB*    
д
@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc*
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter*
parallel_iterations *
is_constant( 
З
@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/NextIteration*
T0*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter*
N* 
_output_shapes
:: 

?training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter*(
_output_shapes
::*
T0

<training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMul_1*
_output_shapes

:*
T0*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/Add*
_output_shapes

:*
T0*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter
т
@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/Switch*
_output_shapes

:*
T0*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter
У
>training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_accConst*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
valueB@*    *
dtype0*
_output_shapes

:@
д
@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc*
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
parallel_iterations *
is_constant( *
_output_shapes

:@*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
З
@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/NextIteration*
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
N* 
_output_shapes
:@: 

?training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*(
_output_shapes
:@:@

<training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_1_grad/MatMul_1*
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
_output_shapes

:@
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/Add*
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
_output_shapes

:@
т
@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
_output_shapes

:@*
T0
Ў
9training/Adam/gradients/lstm_1/strided_slice_9_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_9*
valueB:*
dtype0*
_output_shapes
:
ѕ
Dtraining/Adam/gradients/lstm_1/strided_slice_9_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_9_grad/Shapelstm_1/strided_slice_9/stacklstm_1/strided_slice_9/stack_1lstm_1/strided_slice_9/stack_2Atraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_3*)
_class
loc:@lstm_1/strided_slice_9*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0
Е
9training/Adam/gradients/lstm_1/strided_slice_5_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_5*
valueB"      *
dtype0*
_output_shapes
:
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_5_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_5_grad/Shapelstm_1/strided_slice_5/stacklstm_1/strided_slice_5/stack_1lstm_1/strided_slice_5/stack_2@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_3*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:*
T0*
Index0*)
_class
loc:@lstm_1/strided_slice_5
ѕ
training/Adam/gradients/AddN_8AddN9training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul9training/Adam/gradients/lstm_1/while/MatMul_2_grad/MatMul9training/Adam/gradients/lstm_1/while/MatMul_1_grad/MatMul7training/Adam/gradients/lstm_1/while/MatMul_grad/MatMul*
T0*(
_class
loc:@lstm_1/while/MatMul_3*
N*'
_output_shapes
:џџџџџџџџџ@
§
]training/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3ctraining/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enteretraining/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^training/Adam/gradients/Sub*\
_classR
P#loc:@lstm_1/while/TensorArrayReadV3)loc:@lstm_1/while/TensorArrayReadV3/Enter*#
sourcetraining/Adam/gradients*
_output_shapes

:: 
ї
ctraining/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlstm_1/TensorArray_1*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*\
_classR
P#loc:@lstm_1/while/TensorArrayReadV3)loc:@lstm_1/while/TensorArrayReadV3/Enter*
parallel_iterations 
Ђ
etraining/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterAlstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*
_output_shapes
: *B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*\
_classR
P#loc:@lstm_1/while/TensorArrayReadV3)loc:@lstm_1/while/TensorArrayReadV3/Enter*
parallel_iterations 
Ћ
Ytraining/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityetraining/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^^training/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*\
_classR
P#loc:@lstm_1/while/TensorArrayReadV3)loc:@lstm_1/while/TensorArrayReadV3/Enter*
_output_shapes
: 

_training/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3]training/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3jtraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2training/Adam/gradients/AddN_8Ytraining/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0*1
_class'
%#loc:@lstm_1/while/TensorArrayReadV3*
_output_shapes
: 
П
<training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_accConst*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
valueB@*    *
dtype0*
_output_shapes

:@
Ю
>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_1Enter<training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc*
_output_shapes

:@*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
parallel_iterations *
is_constant( 
Џ
>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_2Merge>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_1Dtraining/Adam/gradients/lstm_1/while/MatMul/Enter_grad/NextIteration*
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
N* 
_output_shapes
:@: 

=training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/SwitchSwitch>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*(
_output_shapes
:@:@*
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter

:training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/AddAdd?training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/Switch:19training/Adam/gradients/lstm_1/while/MatMul_grad/MatMul_1*
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
_output_shapes

:@
ш
Dtraining/Adam/gradients/lstm_1/while/MatMul/Enter_grad/NextIterationNextIteration:training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/Add*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
_output_shapes

:@*
T0
м
>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_3Exit=training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/Switch*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
_output_shapes

:@*
T0
Ў
9training/Adam/gradients/lstm_1/strided_slice_8_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_8*
valueB:*
dtype0*
_output_shapes
:
ѓ
Dtraining/Adam/gradients/lstm_1/strided_slice_8_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_8_grad/Shapelstm_1/strided_slice_8/stacklstm_1/strided_slice_8/stack_1lstm_1/strided_slice_8/stack_2?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_3*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0*)
_class
loc:@lstm_1/strided_slice_8
Ю
Btraining/Adam/gradients/lstm_1/while/Switch_3_grad_1/NextIterationNextIterationtraining/Adam/gradients/AddN_7*'
_output_shapes
:џџџџџџџџџ*
T0*'
_class
loc:@lstm_1/while/Merge_3
Е
9training/Adam/gradients/lstm_1/strided_slice_4_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_4*
valueB"      *
dtype0*
_output_shapes
:
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_4_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_4_grad/Shapelstm_1/strided_slice_4/stacklstm_1/strided_slice_4/stack_1lstm_1/strided_slice_4/stack_2@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_3*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:*
Index0*
T0*)
_class
loc:@lstm_1/strided_slice_4
Е
9training/Adam/gradients/lstm_1/strided_slice_1_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_1*
valueB"@      *
dtype0*
_output_shapes
:
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_1_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_1_grad/Shapelstm_1/strided_slice_1/stacklstm_1/strided_slice_1/stack_1lstm_1/strided_slice_1/stack_2@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_3*)
_class
loc:@lstm_1/strided_slice_1*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:@*
Index0*
T0
Щ
Itraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_accConst*9
_class/
-+loc:@lstm_1/while/TensorArrayReadV3/Enter_1*
valueB
 *    *
dtype0*
_output_shapes
: 
э
Ktraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterItraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc*
is_constant( *
_output_shapes
: *B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*9
_class/
-+loc:@lstm_1/while/TensorArrayReadV3/Enter_1*
parallel_iterations 
л
Ktraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeKtraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Qtraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
N*
_output_shapes
: : *
T0*9
_class/
-+loc:@lstm_1/while/TensorArrayReadV3/Enter_1
Ђ
Jtraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchKtraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*9
_class/
-+loc:@lstm_1/while/TensorArrayReadV3/Enter_1*
_output_shapes
: : 
й
Gtraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/AddAddLtraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/Switch:1_training/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0*9
_class/
-+loc:@lstm_1/while/TensorArrayReadV3/Enter_1*
_output_shapes
: 

Qtraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIterationGtraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/Add*
_output_shapes
: *
T0*9
_class/
-+loc:@lstm_1/while/TensorArrayReadV3/Enter_1
ћ
Ktraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitJtraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/Switch*
_output_shapes
: *
T0*9
_class/
-+loc:@lstm_1/while/TensorArrayReadV3/Enter_1
Б
7training/Adam/gradients/lstm_1/strided_slice_grad/ShapeConst*'
_class
loc:@lstm_1/strided_slice*
valueB"@      *
dtype0*
_output_shapes
:
ъ
Btraining/Adam/gradients/lstm_1/strided_slice_grad/StridedSliceGradStridedSliceGrad7training/Adam/gradients/lstm_1/strided_slice_grad/Shapelstm_1/strided_slice/stacklstm_1/strided_slice/stack_1lstm_1/strided_slice/stack_2>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_3*
end_mask*
_output_shapes

:@*
Index0*
T0*'
_class
loc:@lstm_1/strided_slice*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask 

training/Adam/gradients/AddN_9AddNEtraining/Adam/gradients/lstm_1/strided_slice_11_grad/StridedSliceGradEtraining/Adam/gradients/lstm_1/strided_slice_10_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_9_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_8_grad/StridedSliceGrad*
T0**
_class 
loc:@lstm_1/strided_slice_11*
N*
_output_shapes
:

training/Adam/gradients/AddN_10AddNDtraining/Adam/gradients/lstm_1/strided_slice_7_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_6_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_5_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_4_grad/StridedSliceGrad*
T0*)
_class
loc:@lstm_1/strided_slice_7*
N*
_output_shapes

:
ќ
training/Adam/gradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lstm_1/TensorArray_1Ktraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*>
_class4
2loc:@lstm_1/TensorArray_1loc:@lstm_1/transpose*#
sourcetraining/Adam/gradients*
_output_shapes

:: 
К
|training/Adam/gradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityKtraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3^training/Adam/gradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*>
_class4
2loc:@lstm_1/TensorArray_1loc:@lstm_1/transpose*
_output_shapes
: *
T0
Ђ
rtraining/Adam/gradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3training/Adam/gradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3lstm_1/TensorArrayUnstack/range|training/Adam/gradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*#
_class
loc:@lstm_1/transpose*
dtype0*,
_output_shapes
:Ѕџџџџџџџџџ@*
element_shape:

training/Adam/gradients/AddN_11AddNDtraining/Adam/gradients/lstm_1/strided_slice_3_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_2_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_1_grad/StridedSliceGradBtraining/Adam/gradients/lstm_1/strided_slice_grad/StridedSliceGrad*
T0*)
_class
loc:@lstm_1/strided_slice_3*
N*
_output_shapes

:@
Е
?training/Adam/gradients/lstm_1/transpose_grad/InvertPermutationInvertPermutationlstm_1/transpose/perm*
T0*#
_class
loc:@lstm_1/transpose*
_output_shapes
:
т
7training/Adam/gradients/lstm_1/transpose_grad/transpose	Transposertraining/Adam/gradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3?training/Adam/gradients/lstm_1/transpose_grad/InvertPermutation*,
_output_shapes
:џџџџџџџџџЅ@*
Tperm0*
T0*#
_class
loc:@lstm_1/transpose
р
3training/Adam/gradients/conv1d_2/Relu_grad/ReluGradReluGrad7training/Adam/gradients/lstm_1/transpose_grad/transposeconv1d_2/Relu*
T0* 
_class
loc:@conv1d_2/Relu*,
_output_shapes
:џџџџџџџџџЅ@
Ќ
/training/Adam/gradients/conv1d_2/add_grad/ShapeShapeconv1d_2/convolution/Squeeze*
T0*
_class
loc:@conv1d_2/add*
out_type0*
_output_shapes
:
Ї
1training/Adam/gradients/conv1d_2/add_grad/Shape_1Const*
_output_shapes
:*
_class
loc:@conv1d_2/add*!
valueB"      @   *
dtype0

?training/Adam/gradients/conv1d_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs/training/Adam/gradients/conv1d_2/add_grad/Shape1training/Adam/gradients/conv1d_2/add_grad/Shape_1*
T0*
_class
loc:@conv1d_2/add*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

-training/Adam/gradients/conv1d_2/add_grad/SumSum3training/Adam/gradients/conv1d_2/Relu_grad/ReluGrad?training/Adam/gradients/conv1d_2/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_class
loc:@conv1d_2/add*
_output_shapes
:

1training/Adam/gradients/conv1d_2/add_grad/ReshapeReshape-training/Adam/gradients/conv1d_2/add_grad/Sum/training/Adam/gradients/conv1d_2/add_grad/Shape*
T0*
_class
loc:@conv1d_2/add*
Tshape0*,
_output_shapes
:џџџџџџџџџЅ@

/training/Adam/gradients/conv1d_2/add_grad/Sum_1Sum3training/Adam/gradients/conv1d_2/Relu_grad/ReluGradAtraining/Adam/gradients/conv1d_2/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*
_class
loc:@conv1d_2/add
ў
3training/Adam/gradients/conv1d_2/add_grad/Reshape_1Reshape/training/Adam/gradients/conv1d_2/add_grad/Sum_11training/Adam/gradients/conv1d_2/add_grad/Shape_1*"
_output_shapes
:@*
T0*
_class
loc:@conv1d_2/add*
Tshape0
Ы
?training/Adam/gradients/conv1d_2/convolution/Squeeze_grad/ShapeShapeconv1d_2/convolution/Conv2D*
T0*/
_class%
#!loc:@conv1d_2/convolution/Squeeze*
out_type0*
_output_shapes
:
К
Atraining/Adam/gradients/conv1d_2/convolution/Squeeze_grad/ReshapeReshape1training/Adam/gradients/conv1d_2/add_grad/Reshape?training/Adam/gradients/conv1d_2/convolution/Squeeze_grad/Shape*
T0*/
_class%
#!loc:@conv1d_2/convolution/Squeeze*
Tshape0*0
_output_shapes
:џџџџџџџџџЅ@
Ђ
3training/Adam/gradients/conv1d_2/Reshape_grad/ShapeConst*#
_class
loc:@conv1d_2/Reshape*
valueB:@*
dtype0*
_output_shapes
:

5training/Adam/gradients/conv1d_2/Reshape_grad/ReshapeReshape3training/Adam/gradients/conv1d_2/add_grad/Reshape_13training/Adam/gradients/conv1d_2/Reshape_grad/Shape*#
_class
loc:@conv1d_2/Reshape*
Tshape0*
_output_shapes
:@*
T0

?training/Adam/gradients/conv1d_2/convolution/Conv2D_grad/ShapeNShapeNconv1d_2/convolution/ExpandDims!conv1d_2/convolution/ExpandDims_1*
T0*.
_class$
" loc:@conv1d_2/convolution/Conv2D*
out_type0*
N* 
_output_shapes
::
ф
Ltraining/Adam/gradients/conv1d_2/convolution/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput?training/Adam/gradients/conv1d_2/convolution/Conv2D_grad/ShapeN!conv1d_2/convolution/ExpandDims_1Atraining/Adam/gradients/conv1d_2/convolution/Squeeze_grad/Reshape*0
_output_shapes
:џџџџџџџџџЌ*
	dilations
*
T0*.
_class$
" loc:@conv1d_2/convolution/Conv2D*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
м
Mtraining/Adam/gradients/conv1d_2/convolution/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterconv1d_2/convolution/ExpandDimsAtraining/Adam/gradients/conv1d_2/convolution/Conv2D_grad/ShapeN:1Atraining/Adam/gradients/conv1d_2/convolution/Squeeze_grad/Reshape*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:@*
	dilations
*
T0*.
_class$
" loc:@conv1d_2/convolution/Conv2D
У
Btraining/Adam/gradients/conv1d_2/convolution/ExpandDims_grad/ShapeShapeconv1d_1/Relu*
T0*2
_class(
&$loc:@conv1d_2/convolution/ExpandDims*
out_type0*
_output_shapes
:
к
Dtraining/Adam/gradients/conv1d_2/convolution/ExpandDims_grad/ReshapeReshapeLtraining/Adam/gradients/conv1d_2/convolution/Conv2D_grad/Conv2DBackpropInputBtraining/Adam/gradients/conv1d_2/convolution/ExpandDims_grad/Shape*
T0*2
_class(
&$loc:@conv1d_2/convolution/ExpandDims*
Tshape0*,
_output_shapes
:џџџџџџџџџЌ
Я
Dtraining/Adam/gradients/conv1d_2/convolution/ExpandDims_1_grad/ShapeConst*
dtype0*
_output_shapes
:*4
_class*
(&loc:@conv1d_2/convolution/ExpandDims_1*!
valueB"      @   
з
Ftraining/Adam/gradients/conv1d_2/convolution/ExpandDims_1_grad/ReshapeReshapeMtraining/Adam/gradients/conv1d_2/convolution/Conv2D_grad/Conv2DBackpropFilterDtraining/Adam/gradients/conv1d_2/convolution/ExpandDims_1_grad/Shape*
T0*4
_class*
(&loc:@conv1d_2/convolution/ExpandDims_1*
Tshape0*"
_output_shapes
:@
э
3training/Adam/gradients/conv1d_1/Relu_grad/ReluGradReluGradDtraining/Adam/gradients/conv1d_2/convolution/ExpandDims_grad/Reshapeconv1d_1/Relu*
T0* 
_class
loc:@conv1d_1/Relu*,
_output_shapes
:џџџџџџџџџЌ
Ќ
/training/Adam/gradients/conv1d_1/add_grad/ShapeShapeconv1d_1/convolution/Squeeze*
T0*
_class
loc:@conv1d_1/add*
out_type0*
_output_shapes
:
Ї
1training/Adam/gradients/conv1d_1/add_grad/Shape_1Const*
_class
loc:@conv1d_1/add*!
valueB"         *
dtype0*
_output_shapes
:

?training/Adam/gradients/conv1d_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs/training/Adam/gradients/conv1d_1/add_grad/Shape1training/Adam/gradients/conv1d_1/add_grad/Shape_1*
T0*
_class
loc:@conv1d_1/add*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

-training/Adam/gradients/conv1d_1/add_grad/SumSum3training/Adam/gradients/conv1d_1/Relu_grad/ReluGrad?training/Adam/gradients/conv1d_1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*
_class
loc:@conv1d_1/add

1training/Adam/gradients/conv1d_1/add_grad/ReshapeReshape-training/Adam/gradients/conv1d_1/add_grad/Sum/training/Adam/gradients/conv1d_1/add_grad/Shape*
T0*
_class
loc:@conv1d_1/add*
Tshape0*,
_output_shapes
:џџџџџџџџџЌ

/training/Adam/gradients/conv1d_1/add_grad/Sum_1Sum3training/Adam/gradients/conv1d_1/Relu_grad/ReluGradAtraining/Adam/gradients/conv1d_1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_class
loc:@conv1d_1/add*
_output_shapes
:
ў
3training/Adam/gradients/conv1d_1/add_grad/Reshape_1Reshape/training/Adam/gradients/conv1d_1/add_grad/Sum_11training/Adam/gradients/conv1d_1/add_grad/Shape_1*
T0*
_class
loc:@conv1d_1/add*
Tshape0*"
_output_shapes
:
Ы
?training/Adam/gradients/conv1d_1/convolution/Squeeze_grad/ShapeShapeconv1d_1/convolution/Conv2D*
T0*/
_class%
#!loc:@conv1d_1/convolution/Squeeze*
out_type0*
_output_shapes
:
К
Atraining/Adam/gradients/conv1d_1/convolution/Squeeze_grad/ReshapeReshape1training/Adam/gradients/conv1d_1/add_grad/Reshape?training/Adam/gradients/conv1d_1/convolution/Squeeze_grad/Shape*
T0*/
_class%
#!loc:@conv1d_1/convolution/Squeeze*
Tshape0*0
_output_shapes
:џџџџџџџџџЌ
Ђ
3training/Adam/gradients/conv1d_1/Reshape_grad/ShapeConst*#
_class
loc:@conv1d_1/Reshape*
valueB:*
dtype0*
_output_shapes
:

5training/Adam/gradients/conv1d_1/Reshape_grad/ReshapeReshape3training/Adam/gradients/conv1d_1/add_grad/Reshape_13training/Adam/gradients/conv1d_1/Reshape_grad/Shape*
T0*#
_class
loc:@conv1d_1/Reshape*
Tshape0*
_output_shapes
:

?training/Adam/gradients/conv1d_1/convolution/Conv2D_grad/ShapeNShapeNconv1d_1/convolution/ExpandDims!conv1d_1/convolution/ExpandDims_1* 
_output_shapes
::*
T0*.
_class$
" loc:@conv1d_1/convolution/Conv2D*
out_type0*
N
ф
Ltraining/Adam/gradients/conv1d_1/convolution/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput?training/Adam/gradients/conv1d_1/convolution/Conv2D_grad/ShapeN!conv1d_1/convolution/ExpandDims_1Atraining/Adam/gradients/conv1d_1/convolution/Squeeze_grad/Reshape*
paddingVALID*0
_output_shapes
:џџџџџџџџџЏ*
	dilations
*
T0*.
_class$
" loc:@conv1d_1/convolution/Conv2D*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
м
Mtraining/Adam/gradients/conv1d_1/convolution/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterconv1d_1/convolution/ExpandDimsAtraining/Adam/gradients/conv1d_1/convolution/Conv2D_grad/ShapeN:1Atraining/Adam/gradients/conv1d_1/convolution/Squeeze_grad/Reshape*
paddingVALID*&
_output_shapes
:*
	dilations
*
T0*.
_class$
" loc:@conv1d_1/convolution/Conv2D*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Я
Dtraining/Adam/gradients/conv1d_1/convolution/ExpandDims_1_grad/ShapeConst*
_output_shapes
:*4
_class*
(&loc:@conv1d_1/convolution/ExpandDims_1*!
valueB"         *
dtype0
з
Ftraining/Adam/gradients/conv1d_1/convolution/ExpandDims_1_grad/ReshapeReshapeMtraining/Adam/gradients/conv1d_1/convolution/Conv2D_grad/Conv2DBackpropFilterDtraining/Adam/gradients/conv1d_1/convolution/ExpandDims_1_grad/Shape*
T0*4
_class*
(&loc:@conv1d_1/convolution/ExpandDims_1*
Tshape0*"
_output_shapes
:
_
training/Adam/AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
Ќ
training/Adam/AssignAdd	AssignAddAdam/iterationstraining/Adam/AssignAdd/value*
use_locking( *
T0	*"
_class
loc:@Adam/iterations*
_output_shapes
: 
p
training/Adam/CastCastAdam/iterations/read*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0	
X
training/Adam/add/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
b
training/Adam/addAddtraining/Adam/Casttraining/Adam/add/y*
T0*
_output_shapes
: 
^
training/Adam/PowPowAdam/beta_2/readtraining/Adam/add*
T0*
_output_shapes
: 
X
training/Adam/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
_output_shapes
: *
T0
X
training/Adam/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_1Const*
valueB
 *  *
dtype0*
_output_shapes
: 
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_1*
T0*
_output_shapes
: 

training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const*
T0*
_output_shapes
: 
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
_output_shapes
: *
T0
`
training/Adam/Pow_1PowAdam/beta_1/readtraining/Adam/add*
T0*
_output_shapes
: 
Z
training/Adam/sub_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
g
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
T0*
_output_shapes
: 
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
_output_shapes
: *
T0
^
training/Adam/mulMulAdam/lr/readtraining/Adam/truediv*
T0*
_output_shapes
: 
p
training/Adam/zerosConst*!
valueB*    *
dtype0*"
_output_shapes
:

training/Adam/Variable
VariableV2*
shape:*
shared_name *
dtype0*"
_output_shapes
:*
	container 
е
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*
validate_shape(*"
_output_shapes
:*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable

training/Adam/Variable/readIdentitytraining/Adam/Variable*
T0*)
_class
loc:@training/Adam/Variable*"
_output_shapes
:
b
training/Adam/zeros_1Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_1
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
е
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
T0*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes
:
z
%training/Adam/zeros_2/shape_as_tensorConst*!
valueB"      @   *
dtype0*
_output_shapes
:
`
training/Adam/zeros_2/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
 
training/Adam/zeros_2Fill%training/Adam/zeros_2/shape_as_tensortraining/Adam/zeros_2/Const*
T0*

index_type0*"
_output_shapes
:@

training/Adam/Variable_2
VariableV2*"
_output_shapes
:@*
	container *
shape:@*
shared_name *
dtype0
н
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*"
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(

training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*"
_output_shapes
:@*
T0*+
_class!
loc:@training/Adam/Variable_2
b
training/Adam/zeros_3Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_3
VariableV2*
_output_shapes
:@*
	container *
shape:@*
shared_name *
dtype0
е
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/zeros_3*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*
T0*+
_class!
loc:@training/Adam/Variable_3*
_output_shapes
:@
j
training/Adam/zeros_4Const*
valueB@*    *
dtype0*
_output_shapes

:@

training/Adam/Variable_4
VariableV2*
shape
:@*
shared_name *
dtype0*
_output_shapes

:@*
	container 
й
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/zeros_4*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
_output_shapes

:@

training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*
T0*+
_class!
loc:@training/Adam/Variable_4*
_output_shapes

:@
j
training/Adam/zeros_5Const*
valueB*    *
dtype0*
_output_shapes

:

training/Adam/Variable_5
VariableV2*
shared_name *
dtype0*
_output_shapes

:*
	container *
shape
:
й
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5

training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
T0*+
_class!
loc:@training/Adam/Variable_5*
_output_shapes

:
b
training/Adam/zeros_6Const*
_output_shapes
:*
valueB*    *
dtype0

training/Adam/Variable_6
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
е
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
_output_shapes
:*
T0*+
_class!
loc:@training/Adam/Variable_6
v
%training/Adam/zeros_7/shape_as_tensorConst*
valueB"J     *
dtype0*
_output_shapes
:
`
training/Adam/zeros_7/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_7Fill%training/Adam/zeros_7/shape_as_tensortraining/Adam/zeros_7/Const*
T0*

index_type0* 
_output_shapes
:
Ъ

training/Adam/Variable_7
VariableV2*
shape:
Ъ*
shared_name *
dtype0* 
_output_shapes
:
Ъ*
	container 
л
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/zeros_7*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(* 
_output_shapes
:
Ъ

training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7* 
_output_shapes
:
Ъ*
T0*+
_class!
loc:@training/Adam/Variable_7
d
training/Adam/zeros_8Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_8
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
ж
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*
T0*+
_class!
loc:@training/Adam/Variable_8*
_output_shapes	
:
v
%training/Adam/zeros_9/shape_as_tensorConst*
valueB"      *
dtype0*
_output_shapes
:
`
training/Adam/zeros_9/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_9Fill%training/Adam/zeros_9/shape_as_tensortraining/Adam/zeros_9/Const*
T0*

index_type0* 
_output_shapes
:


training/Adam/Variable_9
VariableV2*
dtype0* 
_output_shapes
:
*
	container *
shape:
*
shared_name 
л
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(* 
_output_shapes
:


training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
T0*+
_class!
loc:@training/Adam/Variable_9* 
_output_shapes
:

e
training/Adam/zeros_10Const*
_output_shapes	
:*
valueB*    *
dtype0

training/Adam/Variable_10
VariableV2*
_output_shapes	
:*
	container *
shape:*
shared_name *
dtype0
к
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*
T0*,
_class"
 loc:@training/Adam/Variable_10*
_output_shapes	
:
m
training/Adam/zeros_11Const*
valueB	*    *
dtype0*
_output_shapes
:	

training/Adam/Variable_11
VariableV2*
shared_name *
dtype0*
_output_shapes
:	*
	container *
shape:	
о
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/zeros_11*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:	

training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*
T0*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
:	
c
training/Adam/zeros_12Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_12
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
й
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
T0*,
_class"
 loc:@training/Adam/Variable_12*
_output_shapes
:
s
training/Adam/zeros_13Const*
dtype0*"
_output_shapes
:*!
valueB*    

training/Adam/Variable_13
VariableV2*
shape:*
shared_name *
dtype0*"
_output_shapes
:*
	container 
с
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*"
_output_shapes
:
 
training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*"
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_13
c
training/Adam/zeros_14Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_14
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
й
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14

training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
T0*,
_class"
 loc:@training/Adam/Variable_14*
_output_shapes
:
{
&training/Adam/zeros_15/shape_as_tensorConst*!
valueB"      @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_15/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ѓ
training/Adam/zeros_15Fill&training/Adam/zeros_15/shape_as_tensortraining/Adam/zeros_15/Const*"
_output_shapes
:@*
T0*

index_type0

training/Adam/Variable_15
VariableV2*
shape:@*
shared_name *
dtype0*"
_output_shapes
:@*
	container 
с
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/zeros_15*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*"
_output_shapes
:@*
use_locking(
 
training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*
T0*,
_class"
 loc:@training/Adam/Variable_15*"
_output_shapes
:@
c
training/Adam/zeros_16Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_16
VariableV2*
shared_name *
dtype0*
_output_shapes
:@*
	container *
shape:@
й
 training/Adam/Variable_16/AssignAssigntraining/Adam/Variable_16training/Adam/zeros_16*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(*
_output_shapes
:@*
use_locking(

training/Adam/Variable_16/readIdentitytraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*
_output_shapes
:@*
T0
k
training/Adam/zeros_17Const*
valueB@*    *
dtype0*
_output_shapes

:@

training/Adam/Variable_17
VariableV2*
shape
:@*
shared_name *
dtype0*
_output_shapes

:@*
	container 
н
 training/Adam/Variable_17/AssignAssigntraining/Adam/Variable_17training/Adam/zeros_17*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17

training/Adam/Variable_17/readIdentitytraining/Adam/Variable_17*
T0*,
_class"
 loc:@training/Adam/Variable_17*
_output_shapes

:@
k
training/Adam/zeros_18Const*
valueB*    *
dtype0*
_output_shapes

:

training/Adam/Variable_18
VariableV2*
shape
:*
shared_name *
dtype0*
_output_shapes

:*
	container 
н
 training/Adam/Variable_18/AssignAssigntraining/Adam/Variable_18training/Adam/zeros_18*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*
_output_shapes

:

training/Adam/Variable_18/readIdentitytraining/Adam/Variable_18*
T0*,
_class"
 loc:@training/Adam/Variable_18*
_output_shapes

:
c
training/Adam/zeros_19Const*
dtype0*
_output_shapes
:*
valueB*    

training/Adam/Variable_19
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
й
 training/Adam/Variable_19/AssignAssigntraining/Adam/Variable_19training/Adam/zeros_19*
T0*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_19/readIdentitytraining/Adam/Variable_19*
T0*,
_class"
 loc:@training/Adam/Variable_19*
_output_shapes
:
w
&training/Adam/zeros_20/shape_as_tensorConst*
valueB"J     *
dtype0*
_output_shapes
:
a
training/Adam/zeros_20/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ё
training/Adam/zeros_20Fill&training/Adam/zeros_20/shape_as_tensortraining/Adam/zeros_20/Const*
T0*

index_type0* 
_output_shapes
:
Ъ

training/Adam/Variable_20
VariableV2*
dtype0* 
_output_shapes
:
Ъ*
	container *
shape:
Ъ*
shared_name 
п
 training/Adam/Variable_20/AssignAssigntraining/Adam/Variable_20training/Adam/zeros_20*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20*
validate_shape(* 
_output_shapes
:
Ъ

training/Adam/Variable_20/readIdentitytraining/Adam/Variable_20*
T0*,
_class"
 loc:@training/Adam/Variable_20* 
_output_shapes
:
Ъ
e
training/Adam/zeros_21Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_21
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
к
 training/Adam/Variable_21/AssignAssigntraining/Adam/Variable_21training/Adam/zeros_21*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_21*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_21/readIdentitytraining/Adam/Variable_21*
_output_shapes	
:*
T0*,
_class"
 loc:@training/Adam/Variable_21
w
&training/Adam/zeros_22/shape_as_tensorConst*
valueB"      *
dtype0*
_output_shapes
:
a
training/Adam/zeros_22/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ё
training/Adam/zeros_22Fill&training/Adam/zeros_22/shape_as_tensortraining/Adam/zeros_22/Const*
T0*

index_type0* 
_output_shapes
:


training/Adam/Variable_22
VariableV2*
shape:
*
shared_name *
dtype0* 
_output_shapes
:
*
	container 
п
 training/Adam/Variable_22/AssignAssigntraining/Adam/Variable_22training/Adam/zeros_22*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

training/Adam/Variable_22/readIdentitytraining/Adam/Variable_22*
T0*,
_class"
 loc:@training/Adam/Variable_22* 
_output_shapes
:

e
training/Adam/zeros_23Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_23
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
к
 training/Adam/Variable_23/AssignAssigntraining/Adam/Variable_23training/Adam/zeros_23*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_23*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_23/readIdentitytraining/Adam/Variable_23*
T0*,
_class"
 loc:@training/Adam/Variable_23*
_output_shapes	
:
m
training/Adam/zeros_24Const*
valueB	*    *
dtype0*
_output_shapes
:	

training/Adam/Variable_24
VariableV2*
shared_name *
dtype0*
_output_shapes
:	*
	container *
shape:	
о
 training/Adam/Variable_24/AssignAssigntraining/Adam/Variable_24training/Adam/zeros_24*,
_class"
 loc:@training/Adam/Variable_24*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0

training/Adam/Variable_24/readIdentitytraining/Adam/Variable_24*
T0*,
_class"
 loc:@training/Adam/Variable_24*
_output_shapes
:	
c
training/Adam/zeros_25Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_25
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
й
 training/Adam/Variable_25/AssignAssigntraining/Adam/Variable_25training/Adam/zeros_25*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_25*
validate_shape(*
_output_shapes
:

training/Adam/Variable_25/readIdentitytraining/Adam/Variable_25*
T0*,
_class"
 loc:@training/Adam/Variable_25*
_output_shapes
:
p
&training/Adam/zeros_26/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_26/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_26Fill&training/Adam/zeros_26/shape_as_tensortraining/Adam/zeros_26/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_26
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
й
 training/Adam/Variable_26/AssignAssigntraining/Adam/Variable_26training/Adam/zeros_26*,
_class"
 loc:@training/Adam/Variable_26*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_26/readIdentitytraining/Adam/Variable_26*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_26
p
&training/Adam/zeros_27/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_27/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_27Fill&training/Adam/zeros_27/shape_as_tensortraining/Adam/zeros_27/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_27
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
й
 training/Adam/Variable_27/AssignAssigntraining/Adam/Variable_27training/Adam/zeros_27*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_27

training/Adam/Variable_27/readIdentitytraining/Adam/Variable_27*
T0*,
_class"
 loc:@training/Adam/Variable_27*
_output_shapes
:
p
&training/Adam/zeros_28/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_28/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_28Fill&training/Adam/zeros_28/shape_as_tensortraining/Adam/zeros_28/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_28
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
й
 training/Adam/Variable_28/AssignAssigntraining/Adam/Variable_28training/Adam/zeros_28*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_28*
validate_shape(*
_output_shapes
:

training/Adam/Variable_28/readIdentitytraining/Adam/Variable_28*
T0*,
_class"
 loc:@training/Adam/Variable_28*
_output_shapes
:
p
&training/Adam/zeros_29/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_29/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_29Fill&training/Adam/zeros_29/shape_as_tensortraining/Adam/zeros_29/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_29
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
й
 training/Adam/Variable_29/AssignAssigntraining/Adam/Variable_29training/Adam/zeros_29*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_29*
validate_shape(*
_output_shapes
:

training/Adam/Variable_29/readIdentitytraining/Adam/Variable_29*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_29
p
&training/Adam/zeros_30/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_30/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_30Fill&training/Adam/zeros_30/shape_as_tensortraining/Adam/zeros_30/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_30
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
й
 training/Adam/Variable_30/AssignAssigntraining/Adam/Variable_30training/Adam/zeros_30*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_30*
validate_shape(*
_output_shapes
:

training/Adam/Variable_30/readIdentitytraining/Adam/Variable_30*
T0*,
_class"
 loc:@training/Adam/Variable_30*
_output_shapes
:
p
&training/Adam/zeros_31/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_31/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_31Fill&training/Adam/zeros_31/shape_as_tensortraining/Adam/zeros_31/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_31
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
й
 training/Adam/Variable_31/AssignAssigntraining/Adam/Variable_31training/Adam/zeros_31*
T0*,
_class"
 loc:@training/Adam/Variable_31*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_31/readIdentitytraining/Adam/Variable_31*
T0*,
_class"
 loc:@training/Adam/Variable_31*
_output_shapes
:
p
&training/Adam/zeros_32/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_32/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_32Fill&training/Adam/zeros_32/shape_as_tensortraining/Adam/zeros_32/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_32
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
й
 training/Adam/Variable_32/AssignAssigntraining/Adam/Variable_32training/Adam/zeros_32*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_32*
validate_shape(*
_output_shapes
:

training/Adam/Variable_32/readIdentitytraining/Adam/Variable_32*
T0*,
_class"
 loc:@training/Adam/Variable_32*
_output_shapes
:
p
&training/Adam/zeros_33/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_33/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_33Fill&training/Adam/zeros_33/shape_as_tensortraining/Adam/zeros_33/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_33
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
й
 training/Adam/Variable_33/AssignAssigntraining/Adam/Variable_33training/Adam/zeros_33*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_33*
validate_shape(*
_output_shapes
:

training/Adam/Variable_33/readIdentitytraining/Adam/Variable_33*
T0*,
_class"
 loc:@training/Adam/Variable_33*
_output_shapes
:
p
&training/Adam/zeros_34/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_34/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_34Fill&training/Adam/zeros_34/shape_as_tensortraining/Adam/zeros_34/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_34
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
й
 training/Adam/Variable_34/AssignAssigntraining/Adam/Variable_34training/Adam/zeros_34*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_34*
validate_shape(*
_output_shapes
:

training/Adam/Variable_34/readIdentitytraining/Adam/Variable_34*
T0*,
_class"
 loc:@training/Adam/Variable_34*
_output_shapes
:
p
&training/Adam/zeros_35/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_35/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_35Fill&training/Adam/zeros_35/shape_as_tensortraining/Adam/zeros_35/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_35
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
й
 training/Adam/Variable_35/AssignAssigntraining/Adam/Variable_35training/Adam/zeros_35*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_35*
validate_shape(*
_output_shapes
:

training/Adam/Variable_35/readIdentitytraining/Adam/Variable_35*
T0*,
_class"
 loc:@training/Adam/Variable_35*
_output_shapes
:
p
&training/Adam/zeros_36/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_36/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_36Fill&training/Adam/zeros_36/shape_as_tensortraining/Adam/zeros_36/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_36
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
й
 training/Adam/Variable_36/AssignAssigntraining/Adam/Variable_36training/Adam/zeros_36*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_36*
validate_shape(*
_output_shapes
:

training/Adam/Variable_36/readIdentitytraining/Adam/Variable_36*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_36
p
&training/Adam/zeros_37/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_37/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_37Fill&training/Adam/zeros_37/shape_as_tensortraining/Adam/zeros_37/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_37
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
й
 training/Adam/Variable_37/AssignAssigntraining/Adam/Variable_37training/Adam/zeros_37*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_37*
validate_shape(*
_output_shapes
:

training/Adam/Variable_37/readIdentitytraining/Adam/Variable_37*
T0*,
_class"
 loc:@training/Adam/Variable_37*
_output_shapes
:
p
&training/Adam/zeros_38/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_38/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_38Fill&training/Adam/zeros_38/shape_as_tensortraining/Adam/zeros_38/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_38
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
й
 training/Adam/Variable_38/AssignAssigntraining/Adam/Variable_38training/Adam/zeros_38*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_38*
validate_shape(*
_output_shapes
:

training/Adam/Variable_38/readIdentitytraining/Adam/Variable_38*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_38
v
training/Adam/mul_1MulAdam/beta_1/readtraining/Adam/Variable/read*
T0*"
_output_shapes
:
Z
training/Adam/sub_2/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
d
training/Adam/sub_2Subtraining/Adam/sub_2/xAdam/beta_1/read*
_output_shapes
: *
T0
Є
training/Adam/mul_2Multraining/Adam/sub_2Ftraining/Adam/gradients/conv1d_1/convolution/ExpandDims_1_grad/Reshape*"
_output_shapes
:*
T0
q
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*
T0*"
_output_shapes
:
y
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_13/read*
T0*"
_output_shapes
:
Z
training/Adam/sub_3/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
d
training/Adam/sub_3Subtraining/Adam/sub_3/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/SquareSquareFtraining/Adam/gradients/conv1d_1/convolution/ExpandDims_1_grad/Reshape*
T0*"
_output_shapes
:
r
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
T0*"
_output_shapes
:
q
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
T0*"
_output_shapes
:
o
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*"
_output_shapes
:*
T0
Z
training/Adam/Const_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_3Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_3*
T0*"
_output_shapes
:

training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_2*
T0*"
_output_shapes
:
h
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*"
_output_shapes
:*
T0
Z
training/Adam/add_3/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
t
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*
T0*"
_output_shapes
:
y
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
T0*"
_output_shapes
:
v
training/Adam/sub_4Subconv1d_1/kernel/readtraining/Adam/truediv_1*
T0*"
_output_shapes
:
Ь
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*"
_output_shapes
:
д
training/Adam/Assign_1Assigntraining/Adam/Variable_13training/Adam/add_2*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*"
_output_shapes
:
Р
training/Adam/Assign_2Assignconv1d_1/kerneltraining/Adam/sub_4*
use_locking(*
T0*"
_class
loc:@conv1d_1/kernel*
validate_shape(*"
_output_shapes
:
p
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
_output_shapes
:*
T0
Z
training/Adam/sub_5/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_5Subtraining/Adam/sub_5/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_7Multraining/Adam/sub_55training/Adam/gradients/conv1d_1/Reshape_grad/Reshape*
_output_shapes
:*
T0
i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
_output_shapes
:*
T0
q
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_14/read*
_output_shapes
:*
T0
Z
training/Adam/sub_6/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_6Subtraining/Adam/sub_6/xAdam/beta_2/read*
T0*
_output_shapes
: 
|
training/Adam/Square_1Square5training/Adam/gradients/conv1d_1/Reshape_grad/Reshape*
T0*
_output_shapes
:
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
_output_shapes
:*
T0
i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
_output_shapes
:*
T0
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes
:
Z
training/Adam/Const_4Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_5Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_5*
T0*
_output_shapes
:

training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_4*
T0*
_output_shapes
:
`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
T0*
_output_shapes
:
Z
training/Adam/add_6/yConst*
_output_shapes
: *
valueB
 *Пж3*
dtype0
l
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
T0*
_output_shapes
:
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
T0*
_output_shapes
:
l
training/Adam/sub_7Subconv1d_1/bias/readtraining/Adam/truediv_2*
T0*
_output_shapes
:
Ъ
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
Ь
training/Adam/Assign_4Assigntraining/Adam/Variable_14training/Adam/add_5*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*
_output_shapes
:
Д
training/Adam/Assign_5Assignconv1d_1/biastraining/Adam/sub_7*
use_locking(*
T0* 
_class
loc:@conv1d_1/bias*
validate_shape(*
_output_shapes
:
y
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*
T0*"
_output_shapes
:@
Z
training/Adam/sub_8/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_8Subtraining/Adam/sub_8/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ѕ
training/Adam/mul_12Multraining/Adam/sub_8Ftraining/Adam/gradients/conv1d_2/convolution/ExpandDims_1_grad/Reshape*
T0*"
_output_shapes
:@
s
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
T0*"
_output_shapes
:@
z
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_15/read*
T0*"
_output_shapes
:@
Z
training/Adam/sub_9/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_9Subtraining/Adam/sub_9/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_2SquareFtraining/Adam/gradients/conv1d_2/convolution/ExpandDims_1_grad/Reshape*
T0*"
_output_shapes
:@
u
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*"
_output_shapes
:@*
T0
s
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*"
_output_shapes
:@*
T0
p
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*"
_output_shapes
:@*
T0
Z
training/Adam/Const_6Const*
_output_shapes
: *
valueB
 *    *
dtype0
Z
training/Adam/Const_7Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_7*
T0*"
_output_shapes
:@

training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_6*
T0*"
_output_shapes
:@
h
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*"
_output_shapes
:@*
T0
Z
training/Adam/add_9/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
t
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y*
T0*"
_output_shapes
:@
z
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*"
_output_shapes
:@*
T0
w
training/Adam/sub_10Subconv1d_2/kernel/readtraining/Adam/truediv_3*"
_output_shapes
:@*
T0
в
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*"
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(
д
training/Adam/Assign_7Assigntraining/Adam/Variable_15training/Adam/add_8*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*"
_output_shapes
:@*
use_locking(*
T0
С
training/Adam/Assign_8Assignconv1d_2/kerneltraining/Adam/sub_10*"
_class
loc:@conv1d_2/kernel*
validate_shape(*"
_output_shapes
:@*
use_locking(*
T0
q
training/Adam/mul_16MulAdam/beta_1/readtraining/Adam/Variable_3/read*
_output_shapes
:@*
T0
[
training/Adam/sub_11/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_11Subtraining/Adam/sub_11/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_17Multraining/Adam/sub_115training/Adam/gradients/conv1d_2/Reshape_grad/Reshape*
T0*
_output_shapes
:@
l
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
T0*
_output_shapes
:@
r
training/Adam/mul_18MulAdam/beta_2/readtraining/Adam/Variable_16/read*
T0*
_output_shapes
:@
[
training/Adam/sub_12/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_12Subtraining/Adam/sub_12/xAdam/beta_2/read*
T0*
_output_shapes
: 
|
training/Adam/Square_3Square5training/Adam/gradients/conv1d_2/Reshape_grad/Reshape*
T0*
_output_shapes
:@
n
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
T0*
_output_shapes
:@
l
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
T0*
_output_shapes
:@
i
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
T0*
_output_shapes
:@
Z
training/Adam/Const_8Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_9Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_9*
T0*
_output_shapes
:@

training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_8*
T0*
_output_shapes
:@
`
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
T0*
_output_shapes
:@
[
training/Adam/add_12/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
n
training/Adam/add_12Addtraining/Adam/Sqrt_4training/Adam/add_12/y*
T0*
_output_shapes
:@
s
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
_output_shapes
:@*
T0
m
training/Adam/sub_13Subconv1d_2/bias/readtraining/Adam/truediv_4*
T0*
_output_shapes
:@
Ы
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:@
Ю
training/Adam/Assign_10Assigntraining/Adam/Variable_16training/Adam/add_11*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(*
_output_shapes
:@
Ж
training/Adam/Assign_11Assignconv1d_2/biastraining/Adam/sub_13*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@conv1d_2/bias
u
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*
T0*
_output_shapes

:@
[
training/Adam/sub_14/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_14Subtraining/Adam/sub_14/xAdam/beta_1/read*
T0*
_output_shapes
: 
{
training/Adam/mul_22Multraining/Adam/sub_14training/Adam/gradients/AddN_11*
_output_shapes

:@*
T0
p
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
T0*
_output_shapes

:@
v
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_17/read*
T0*
_output_shapes

:@
[
training/Adam/sub_15/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_15Subtraining/Adam/sub_15/xAdam/beta_2/read*
T0*
_output_shapes
: 
j
training/Adam/Square_4Squaretraining/Adam/gradients/AddN_11*
T0*
_output_shapes

:@
r
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
T0*
_output_shapes

:@
p
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
_output_shapes

:@*
T0
m
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
_output_shapes

:@*
T0
[
training/Adam/Const_10Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_11Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_11*
_output_shapes

:@*
T0

training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_10*
T0*
_output_shapes

:@
d
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
T0*
_output_shapes

:@
[
training/Adam/add_15/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
r
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y*
T0*
_output_shapes

:@
w
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
_output_shapes

:@*
T0
q
training/Adam/sub_16Sublstm_1/kernel/readtraining/Adam/truediv_5*
_output_shapes

:@*
T0
а
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
_output_shapes

:@*
use_locking(
в
training/Adam/Assign_13Assigntraining/Adam/Variable_17training/Adam/add_14*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes

:@
К
training/Adam/Assign_14Assignlstm_1/kerneltraining/Adam/sub_16*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0* 
_class
loc:@lstm_1/kernel
u
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read*
T0*
_output_shapes

:
[
training/Adam/sub_17/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_17Subtraining/Adam/sub_17/xAdam/beta_1/read*
T0*
_output_shapes
: 
{
training/Adam/mul_27Multraining/Adam/sub_17training/Adam/gradients/AddN_10*
T0*
_output_shapes

:
p
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
T0*
_output_shapes

:
v
training/Adam/mul_28MulAdam/beta_2/readtraining/Adam/Variable_18/read*
T0*
_output_shapes

:
[
training/Adam/sub_18/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_18Subtraining/Adam/sub_18/xAdam/beta_2/read*
T0*
_output_shapes
: 
j
training/Adam/Square_5Squaretraining/Adam/gradients/AddN_10*
T0*
_output_shapes

:
r
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
T0*
_output_shapes

:
p
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
T0*
_output_shapes

:
m
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
T0*
_output_shapes

:
[
training/Adam/Const_12Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_13Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_13*
T0*
_output_shapes

:

training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_12*
T0*
_output_shapes

:
d
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
T0*
_output_shapes

:
[
training/Adam/add_18/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
r
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y*
T0*
_output_shapes

:
w
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
T0*
_output_shapes

:
{
training/Adam/sub_19Sublstm_1/recurrent_kernel/readtraining/Adam/truediv_6*
T0*
_output_shapes

:
а
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
_output_shapes

:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(
в
training/Adam/Assign_16Assigntraining/Adam/Variable_18training/Adam/add_17*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*
_output_shapes

:*
use_locking(
Ю
training/Adam/Assign_17Assignlstm_1/recurrent_kerneltraining/Adam/sub_19*
use_locking(*
T0**
_class 
loc:@lstm_1/recurrent_kernel*
validate_shape(*
_output_shapes

:
q
training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*
T0*
_output_shapes
:
[
training/Adam/sub_20/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_20Subtraining/Adam/sub_20/xAdam/beta_1/read*
_output_shapes
: *
T0
v
training/Adam/mul_32Multraining/Adam/sub_20training/Adam/gradients/AddN_9*
T0*
_output_shapes
:
l
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
T0*
_output_shapes
:
r
training/Adam/mul_33MulAdam/beta_2/readtraining/Adam/Variable_19/read*
T0*
_output_shapes
:
[
training/Adam/sub_21/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_21Subtraining/Adam/sub_21/xAdam/beta_2/read*
T0*
_output_shapes
: 
e
training/Adam/Square_6Squaretraining/Adam/gradients/AddN_9*
T0*
_output_shapes
:
n
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
_output_shapes
:*
T0
l
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
T0*
_output_shapes
:
i
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
_output_shapes
:*
T0
[
training/Adam/Const_14Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_15Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_15*
T0*
_output_shapes
:

training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_14*
T0*
_output_shapes
:
`
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
T0*
_output_shapes
:
[
training/Adam/add_21/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
n
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*
T0*
_output_shapes
:
s
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
_output_shapes
:*
T0
k
training/Adam/sub_22Sublstm_1/bias/readtraining/Adam/truediv_7*
_output_shapes
:*
T0
Ь
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*
_output_shapes
:*
use_locking(
Ю
training/Adam/Assign_19Assigntraining/Adam/Variable_19training/Adam/add_20*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
В
training/Adam/Assign_20Assignlstm_1/biastraining/Adam/sub_22*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@lstm_1/bias
w
training/Adam/mul_36MulAdam/beta_1/readtraining/Adam/Variable_7/read* 
_output_shapes
:
Ъ*
T0
[
training/Adam/sub_23/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_23Subtraining/Adam/sub_23/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_37Multraining/Adam/sub_234training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
Ъ
r
training/Adam/add_22Addtraining/Adam/mul_36training/Adam/mul_37*
T0* 
_output_shapes
:
Ъ
x
training/Adam/mul_38MulAdam/beta_2/readtraining/Adam/Variable_20/read*
T0* 
_output_shapes
:
Ъ
[
training/Adam/sub_24/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_24Subtraining/Adam/sub_24/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_7Square4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
Ъ
t
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7* 
_output_shapes
:
Ъ*
T0
r
training/Adam/add_23Addtraining/Adam/mul_38training/Adam/mul_39*
T0* 
_output_shapes
:
Ъ
o
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22* 
_output_shapes
:
Ъ*
T0
[
training/Adam/Const_16Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_17Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_17* 
_output_shapes
:
Ъ*
T0

training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_16* 
_output_shapes
:
Ъ*
T0
f
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8* 
_output_shapes
:
Ъ*
T0
[
training/Adam/add_24/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
t
training/Adam/add_24Addtraining/Adam/Sqrt_8training/Adam/add_24/y*
T0* 
_output_shapes
:
Ъ
y
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24*
T0* 
_output_shapes
:
Ъ
t
training/Adam/sub_25Subdense_1/kernel/readtraining/Adam/truediv_8*
T0* 
_output_shapes
:
Ъ
в
training/Adam/Assign_21Assigntraining/Adam/Variable_7training/Adam/add_22*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(* 
_output_shapes
:
Ъ
д
training/Adam/Assign_22Assigntraining/Adam/Variable_20training/Adam/add_23*
validate_shape(* 
_output_shapes
:
Ъ*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20
О
training/Adam/Assign_23Assigndense_1/kerneltraining/Adam/sub_25*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(* 
_output_shapes
:
Ъ
r
training/Adam/mul_41MulAdam/beta_1/readtraining/Adam/Variable_8/read*
_output_shapes	
:*
T0
[
training/Adam/sub_26/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_26Subtraining/Adam/sub_26/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_42Multraining/Adam/sub_268training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
m
training/Adam/add_25Addtraining/Adam/mul_41training/Adam/mul_42*
_output_shapes	
:*
T0
s
training/Adam/mul_43MulAdam/beta_2/readtraining/Adam/Variable_21/read*
_output_shapes	
:*
T0
[
training/Adam/sub_27/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_27Subtraining/Adam/sub_27/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_8Square8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
o
training/Adam/mul_44Multraining/Adam/sub_27training/Adam/Square_8*
_output_shapes	
:*
T0
m
training/Adam/add_26Addtraining/Adam/mul_43training/Adam/mul_44*
T0*
_output_shapes	
:
j
training/Adam/mul_45Multraining/Adam/multraining/Adam/add_25*
T0*
_output_shapes	
:
[
training/Adam/Const_18Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_19Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_9/MinimumMinimumtraining/Adam/add_26training/Adam/Const_19*
T0*
_output_shapes	
:

training/Adam/clip_by_value_9Maximum%training/Adam/clip_by_value_9/Minimumtraining/Adam/Const_18*
T0*
_output_shapes	
:
a
training/Adam/Sqrt_9Sqrttraining/Adam/clip_by_value_9*
_output_shapes	
:*
T0
[
training/Adam/add_27/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
o
training/Adam/add_27Addtraining/Adam/Sqrt_9training/Adam/add_27/y*
_output_shapes	
:*
T0
t
training/Adam/truediv_9RealDivtraining/Adam/mul_45training/Adam/add_27*
T0*
_output_shapes	
:
m
training/Adam/sub_28Subdense_1/bias/readtraining/Adam/truediv_9*
T0*
_output_shapes	
:
Э
training/Adam/Assign_24Assigntraining/Adam/Variable_8training/Adam/add_25*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8
Я
training/Adam/Assign_25Assigntraining/Adam/Variable_21training/Adam/add_26*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_21*
validate_shape(*
_output_shapes	
:
Е
training/Adam/Assign_26Assigndense_1/biastraining/Adam/sub_28*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:
w
training/Adam/mul_46MulAdam/beta_1/readtraining/Adam/Variable_9/read*
T0* 
_output_shapes
:

[
training/Adam/sub_29/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_29Subtraining/Adam/sub_29/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_47Multraining/Adam/sub_294training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

r
training/Adam/add_28Addtraining/Adam/mul_46training/Adam/mul_47*
T0* 
_output_shapes
:

x
training/Adam/mul_48MulAdam/beta_2/readtraining/Adam/Variable_22/read*
T0* 
_output_shapes
:

[
training/Adam/sub_30/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_30Subtraining/Adam/sub_30/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_9Square4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

t
training/Adam/mul_49Multraining/Adam/sub_30training/Adam/Square_9*
T0* 
_output_shapes
:

r
training/Adam/add_29Addtraining/Adam/mul_48training/Adam/mul_49* 
_output_shapes
:
*
T0
o
training/Adam/mul_50Multraining/Adam/multraining/Adam/add_28*
T0* 
_output_shapes
:

[
training/Adam/Const_20Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_21Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_10/MinimumMinimumtraining/Adam/add_29training/Adam/Const_21*
T0* 
_output_shapes
:


training/Adam/clip_by_value_10Maximum&training/Adam/clip_by_value_10/Minimumtraining/Adam/Const_20* 
_output_shapes
:
*
T0
h
training/Adam/Sqrt_10Sqrttraining/Adam/clip_by_value_10*
T0* 
_output_shapes
:

[
training/Adam/add_30/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
u
training/Adam/add_30Addtraining/Adam/Sqrt_10training/Adam/add_30/y*
T0* 
_output_shapes
:

z
training/Adam/truediv_10RealDivtraining/Adam/mul_50training/Adam/add_30*
T0* 
_output_shapes
:

u
training/Adam/sub_31Subdense_2/kernel/readtraining/Adam/truediv_10* 
_output_shapes
:
*
T0
в
training/Adam/Assign_27Assigntraining/Adam/Variable_9training/Adam/add_28*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(* 
_output_shapes
:

д
training/Adam/Assign_28Assigntraining/Adam/Variable_22training/Adam/add_29*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
О
training/Adam/Assign_29Assigndense_2/kerneltraining/Adam/sub_31*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(* 
_output_shapes
:

s
training/Adam/mul_51MulAdam/beta_1/readtraining/Adam/Variable_10/read*
T0*
_output_shapes	
:
[
training/Adam/sub_32/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_32Subtraining/Adam/sub_32/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_52Multraining/Adam/sub_328training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
m
training/Adam/add_31Addtraining/Adam/mul_51training/Adam/mul_52*
T0*
_output_shapes	
:
s
training/Adam/mul_53MulAdam/beta_2/readtraining/Adam/Variable_23/read*
_output_shapes	
:*
T0
[
training/Adam/sub_33/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_33Subtraining/Adam/sub_33/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_10Square8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
p
training/Adam/mul_54Multraining/Adam/sub_33training/Adam/Square_10*
T0*
_output_shapes	
:
m
training/Adam/add_32Addtraining/Adam/mul_53training/Adam/mul_54*
_output_shapes	
:*
T0
j
training/Adam/mul_55Multraining/Adam/multraining/Adam/add_31*
T0*
_output_shapes	
:
[
training/Adam/Const_22Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_23Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_11/MinimumMinimumtraining/Adam/add_32training/Adam/Const_23*
T0*
_output_shapes	
:

training/Adam/clip_by_value_11Maximum&training/Adam/clip_by_value_11/Minimumtraining/Adam/Const_22*
_output_shapes	
:*
T0
c
training/Adam/Sqrt_11Sqrttraining/Adam/clip_by_value_11*
T0*
_output_shapes	
:
[
training/Adam/add_33/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
p
training/Adam/add_33Addtraining/Adam/Sqrt_11training/Adam/add_33/y*
_output_shapes	
:*
T0
u
training/Adam/truediv_11RealDivtraining/Adam/mul_55training/Adam/add_33*
T0*
_output_shapes	
:
n
training/Adam/sub_34Subdense_2/bias/readtraining/Adam/truediv_11*
T0*
_output_shapes	
:
Я
training/Adam/Assign_30Assigntraining/Adam/Variable_10training/Adam/add_31*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10
Я
training/Adam/Assign_31Assigntraining/Adam/Variable_23training/Adam/add_32*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_23*
validate_shape(*
_output_shapes	
:
Е
training/Adam/Assign_32Assigndense_2/biastraining/Adam/sub_34*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes	
:
w
training/Adam/mul_56MulAdam/beta_1/readtraining/Adam/Variable_11/read*
_output_shapes
:	*
T0
[
training/Adam/sub_35/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_35Subtraining/Adam/sub_35/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_57Multraining/Adam/sub_354training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
_output_shapes
:	*
T0
q
training/Adam/add_34Addtraining/Adam/mul_56training/Adam/mul_57*
_output_shapes
:	*
T0
w
training/Adam/mul_58MulAdam/beta_2/readtraining/Adam/Variable_24/read*
_output_shapes
:	*
T0
[
training/Adam/sub_36/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_36Subtraining/Adam/sub_36/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_11Square4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	
t
training/Adam/mul_59Multraining/Adam/sub_36training/Adam/Square_11*
T0*
_output_shapes
:	
q
training/Adam/add_35Addtraining/Adam/mul_58training/Adam/mul_59*
T0*
_output_shapes
:	
n
training/Adam/mul_60Multraining/Adam/multraining/Adam/add_34*
_output_shapes
:	*
T0
[
training/Adam/Const_24Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_25Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_12/MinimumMinimumtraining/Adam/add_35training/Adam/Const_25*
T0*
_output_shapes
:	

training/Adam/clip_by_value_12Maximum&training/Adam/clip_by_value_12/Minimumtraining/Adam/Const_24*
_output_shapes
:	*
T0
g
training/Adam/Sqrt_12Sqrttraining/Adam/clip_by_value_12*
_output_shapes
:	*
T0
[
training/Adam/add_36/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
t
training/Adam/add_36Addtraining/Adam/Sqrt_12training/Adam/add_36/y*
_output_shapes
:	*
T0
y
training/Adam/truediv_12RealDivtraining/Adam/mul_60training/Adam/add_36*
T0*
_output_shapes
:	
t
training/Adam/sub_37Subdense_3/kernel/readtraining/Adam/truediv_12*
T0*
_output_shapes
:	
г
training/Adam/Assign_33Assigntraining/Adam/Variable_11training/Adam/add_34*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:	
г
training/Adam/Assign_34Assigntraining/Adam/Variable_24training/Adam/add_35*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_24*
validate_shape(*
_output_shapes
:	
Н
training/Adam/Assign_35Assigndense_3/kerneltraining/Adam/sub_37*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes
:	
r
training/Adam/mul_61MulAdam/beta_1/readtraining/Adam/Variable_12/read*
T0*
_output_shapes
:
[
training/Adam/sub_38/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_38Subtraining/Adam/sub_38/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_62Multraining/Adam/sub_388training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
l
training/Adam/add_37Addtraining/Adam/mul_61training/Adam/mul_62*
T0*
_output_shapes
:
r
training/Adam/mul_63MulAdam/beta_2/readtraining/Adam/Variable_25/read*
_output_shapes
:*
T0
[
training/Adam/sub_39/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_39Subtraining/Adam/sub_39/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_12Square8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
o
training/Adam/mul_64Multraining/Adam/sub_39training/Adam/Square_12*
T0*
_output_shapes
:
l
training/Adam/add_38Addtraining/Adam/mul_63training/Adam/mul_64*
T0*
_output_shapes
:
i
training/Adam/mul_65Multraining/Adam/multraining/Adam/add_37*
T0*
_output_shapes
:
[
training/Adam/Const_26Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_27Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_13/MinimumMinimumtraining/Adam/add_38training/Adam/Const_27*
T0*
_output_shapes
:

training/Adam/clip_by_value_13Maximum&training/Adam/clip_by_value_13/Minimumtraining/Adam/Const_26*
T0*
_output_shapes
:
b
training/Adam/Sqrt_13Sqrttraining/Adam/clip_by_value_13*
T0*
_output_shapes
:
[
training/Adam/add_39/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
o
training/Adam/add_39Addtraining/Adam/Sqrt_13training/Adam/add_39/y*
T0*
_output_shapes
:
t
training/Adam/truediv_13RealDivtraining/Adam/mul_65training/Adam/add_39*
T0*
_output_shapes
:
m
training/Adam/sub_40Subdense_3/bias/readtraining/Adam/truediv_13*
_output_shapes
:*
T0
Ю
training/Adam/Assign_36Assigntraining/Adam/Variable_12training/Adam/add_37*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*
_output_shapes
:*
use_locking(
Ю
training/Adam/Assign_37Assigntraining/Adam/Variable_25training/Adam/add_38*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_25*
validate_shape(*
_output_shapes
:
Д
training/Adam/Assign_38Assigndense_3/biastraining/Adam/sub_40*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:
Н
training/group_depsNoOp	^loss/mul^metrics/acc/Mean^training/Adam/Assign^training/Adam/AssignAdd^training/Adam/Assign_1^training/Adam/Assign_10^training/Adam/Assign_11^training/Adam/Assign_12^training/Adam/Assign_13^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_18^training/Adam/Assign_19^training/Adam/Assign_2^training/Adam/Assign_20^training/Adam/Assign_21^training/Adam/Assign_22^training/Adam/Assign_23^training/Adam/Assign_24^training/Adam/Assign_25^training/Adam/Assign_26^training/Adam/Assign_27^training/Adam/Assign_28^training/Adam/Assign_29^training/Adam/Assign_3^training/Adam/Assign_30^training/Adam/Assign_31^training/Adam/Assign_32^training/Adam/Assign_33^training/Adam/Assign_34^training/Adam/Assign_35^training/Adam/Assign_36^training/Adam/Assign_37^training/Adam/Assign_38^training/Adam/Assign_4^training/Adam/Assign_5^training/Adam/Assign_6^training/Adam/Assign_7^training/Adam/Assign_8^training/Adam/Assign_9

IsVariableInitializedIsVariableInitializedconv1d_1/kernel*"
_class
loc:@conv1d_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_1IsVariableInitializedconv1d_1/bias*
dtype0*
_output_shapes
: * 
_class
loc:@conv1d_1/bias

IsVariableInitialized_2IsVariableInitializedconv1d_2/kernel*"
_class
loc:@conv1d_2/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_3IsVariableInitializedconv1d_2/bias* 
_class
loc:@conv1d_2/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_4IsVariableInitializedlstm_1/kernel* 
_class
loc:@lstm_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_5IsVariableInitializedlstm_1/recurrent_kernel**
_class 
loc:@lstm_1/recurrent_kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_6IsVariableInitializedlstm_1/bias*
_output_shapes
: *
_class
loc:@lstm_1/bias*
dtype0

IsVariableInitialized_7IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_8IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_9IsVariableInitializeddense_2/kernel*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_10IsVariableInitializeddense_2/bias*
_class
loc:@dense_2/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_11IsVariableInitializeddense_3/kernel*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_12IsVariableInitializeddense_3/bias*
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_13IsVariableInitializedAdam/iterations*"
_class
loc:@Adam/iterations*
dtype0	*
_output_shapes
: 
{
IsVariableInitialized_14IsVariableInitializedAdam/lr*
_output_shapes
: *
_class
loc:@Adam/lr*
dtype0

IsVariableInitialized_15IsVariableInitializedAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_16IsVariableInitializedAdam/beta_2*
dtype0*
_output_shapes
: *
_class
loc:@Adam/beta_2

IsVariableInitialized_17IsVariableInitialized
Adam/decay*
_output_shapes
: *
_class
loc:@Adam/decay*
dtype0

IsVariableInitialized_18IsVariableInitializedtraining/Adam/Variable*
dtype0*
_output_shapes
: *)
_class
loc:@training/Adam/Variable

IsVariableInitialized_19IsVariableInitializedtraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_20IsVariableInitializedtraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0*
_output_shapes
: 

IsVariableInitialized_21IsVariableInitializedtraining/Adam/Variable_3*+
_class!
loc:@training/Adam/Variable_3*
dtype0*
_output_shapes
: 

IsVariableInitialized_22IsVariableInitializedtraining/Adam/Variable_4*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_4

IsVariableInitialized_23IsVariableInitializedtraining/Adam/Variable_5*+
_class!
loc:@training/Adam/Variable_5*
dtype0*
_output_shapes
: 

IsVariableInitialized_24IsVariableInitializedtraining/Adam/Variable_6*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_6*
dtype0

IsVariableInitialized_25IsVariableInitializedtraining/Adam/Variable_7*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_7*
dtype0

IsVariableInitialized_26IsVariableInitializedtraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
dtype0*
_output_shapes
: 

IsVariableInitialized_27IsVariableInitializedtraining/Adam/Variable_9*+
_class!
loc:@training/Adam/Variable_9*
dtype0*
_output_shapes
: 

IsVariableInitialized_28IsVariableInitializedtraining/Adam/Variable_10*,
_class"
 loc:@training/Adam/Variable_10*
dtype0*
_output_shapes
: 

IsVariableInitialized_29IsVariableInitializedtraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
dtype0*
_output_shapes
: 

IsVariableInitialized_30IsVariableInitializedtraining/Adam/Variable_12*,
_class"
 loc:@training/Adam/Variable_12*
dtype0*
_output_shapes
: 

IsVariableInitialized_31IsVariableInitializedtraining/Adam/Variable_13*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_13

IsVariableInitialized_32IsVariableInitializedtraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
dtype0*
_output_shapes
: 

IsVariableInitialized_33IsVariableInitializedtraining/Adam/Variable_15*,
_class"
 loc:@training/Adam/Variable_15*
dtype0*
_output_shapes
: 

IsVariableInitialized_34IsVariableInitializedtraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*
dtype0*
_output_shapes
: 

IsVariableInitialized_35IsVariableInitializedtraining/Adam/Variable_17*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_17*
dtype0

IsVariableInitialized_36IsVariableInitializedtraining/Adam/Variable_18*,
_class"
 loc:@training/Adam/Variable_18*
dtype0*
_output_shapes
: 

IsVariableInitialized_37IsVariableInitializedtraining/Adam/Variable_19*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_19

IsVariableInitialized_38IsVariableInitializedtraining/Adam/Variable_20*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_20*
dtype0

IsVariableInitialized_39IsVariableInitializedtraining/Adam/Variable_21*,
_class"
 loc:@training/Adam/Variable_21*
dtype0*
_output_shapes
: 

IsVariableInitialized_40IsVariableInitializedtraining/Adam/Variable_22*,
_class"
 loc:@training/Adam/Variable_22*
dtype0*
_output_shapes
: 

IsVariableInitialized_41IsVariableInitializedtraining/Adam/Variable_23*,
_class"
 loc:@training/Adam/Variable_23*
dtype0*
_output_shapes
: 

IsVariableInitialized_42IsVariableInitializedtraining/Adam/Variable_24*,
_class"
 loc:@training/Adam/Variable_24*
dtype0*
_output_shapes
: 

IsVariableInitialized_43IsVariableInitializedtraining/Adam/Variable_25*,
_class"
 loc:@training/Adam/Variable_25*
dtype0*
_output_shapes
: 

IsVariableInitialized_44IsVariableInitializedtraining/Adam/Variable_26*,
_class"
 loc:@training/Adam/Variable_26*
dtype0*
_output_shapes
: 

IsVariableInitialized_45IsVariableInitializedtraining/Adam/Variable_27*,
_class"
 loc:@training/Adam/Variable_27*
dtype0*
_output_shapes
: 

IsVariableInitialized_46IsVariableInitializedtraining/Adam/Variable_28*,
_class"
 loc:@training/Adam/Variable_28*
dtype0*
_output_shapes
: 

IsVariableInitialized_47IsVariableInitializedtraining/Adam/Variable_29*,
_class"
 loc:@training/Adam/Variable_29*
dtype0*
_output_shapes
: 

IsVariableInitialized_48IsVariableInitializedtraining/Adam/Variable_30*,
_class"
 loc:@training/Adam/Variable_30*
dtype0*
_output_shapes
: 

IsVariableInitialized_49IsVariableInitializedtraining/Adam/Variable_31*,
_class"
 loc:@training/Adam/Variable_31*
dtype0*
_output_shapes
: 

IsVariableInitialized_50IsVariableInitializedtraining/Adam/Variable_32*,
_class"
 loc:@training/Adam/Variable_32*
dtype0*
_output_shapes
: 

IsVariableInitialized_51IsVariableInitializedtraining/Adam/Variable_33*,
_class"
 loc:@training/Adam/Variable_33*
dtype0*
_output_shapes
: 

IsVariableInitialized_52IsVariableInitializedtraining/Adam/Variable_34*,
_class"
 loc:@training/Adam/Variable_34*
dtype0*
_output_shapes
: 

IsVariableInitialized_53IsVariableInitializedtraining/Adam/Variable_35*,
_class"
 loc:@training/Adam/Variable_35*
dtype0*
_output_shapes
: 

IsVariableInitialized_54IsVariableInitializedtraining/Adam/Variable_36*,
_class"
 loc:@training/Adam/Variable_36*
dtype0*
_output_shapes
: 

IsVariableInitialized_55IsVariableInitializedtraining/Adam/Variable_37*,
_class"
 loc:@training/Adam/Variable_37*
dtype0*
_output_shapes
: 

IsVariableInitialized_56IsVariableInitializedtraining/Adam/Variable_38*,
_class"
 loc:@training/Adam/Variable_38*
dtype0*
_output_shapes
: 
є
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^conv1d_1/bias/Assign^conv1d_1/kernel/Assign^conv1d_2/bias/Assign^conv1d_2/kernel/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^dense_3/bias/Assign^dense_3/kernel/Assign^lstm_1/bias/Assign^lstm_1/kernel/Assign^lstm_1/recurrent_kernel/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign!^training/Adam/Variable_18/Assign!^training/Adam/Variable_19/Assign ^training/Adam/Variable_2/Assign!^training/Adam/Variable_20/Assign!^training/Adam/Variable_21/Assign!^training/Adam/Variable_22/Assign!^training/Adam/Variable_23/Assign!^training/Adam/Variable_24/Assign!^training/Adam/Variable_25/Assign!^training/Adam/Variable_26/Assign!^training/Adam/Variable_27/Assign!^training/Adam/Variable_28/Assign!^training/Adam/Variable_29/Assign ^training/Adam/Variable_3/Assign!^training/Adam/Variable_30/Assign!^training/Adam/Variable_31/Assign!^training/Adam/Variable_32/Assign!^training/Adam/Variable_33/Assign!^training/Adam/Variable_34/Assign!^training/Adam/Variable_35/Assign!^training/Adam/Variable_36/Assign!^training/Adam/Variable_37/Assign!^training/Adam/Variable_38/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign"ь@HЇН     (yY	vіЅкћжAJћ
<ђ;
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignAdd
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

ControlTrigger
ь
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

B
Equal
x"T
y"T
z
"
Ttype:
2	

)
Exit	
data"T
output"T"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
InvertPermutation
x"T
y"T"
Ttype0:
2	
N
IsVariableInitialized
ref"dtype
is_initialized
"
dtypetype
:
Less
x"T
y"T
z
"
Ttype:
2	
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
$

LogicalAnd
x

y

z

!
LoopCond	
input


output

p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
;
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	

M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
)
Rank

input"T

output"	
Ttype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
A

StackPopV2

handle
elem"	elem_type"
	elem_typetype
X
StackPushV2

handle	
elem"T
output"T"	
Ttype"
swap_memorybool( 
S
StackV2
max_size

handle"
	elem_typetype"

stack_namestring 
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 

StridedSliceGrad
shape"Index
begin"Index
end"Index
strides"Index
dy"T
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:
`
TensorArrayGradV3

handle
flow_in
grad_handle
flow_out"
sourcestring
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype
9
TensorArraySizeV3

handle
flow_in
size
о
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring 
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype*1.12.02v1.12.0-rc2-3-ga6d8ffae09Аї
{
conv1d_1_inputPlaceholder*
dtype0*,
_output_shapes
:џџџџџџџџџЏ*!
shape:џџџџџџџџџЏ
r
conv1d_1/random_uniform/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
`
conv1d_1/random_uniform/minConst*
valueB
 *ьбО*
dtype0*
_output_shapes
: 
`
conv1d_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ьб>
Ў
%conv1d_1/random_uniform/RandomUniformRandomUniformconv1d_1/random_uniform/shape*
seedБџх)*
T0*
dtype0*"
_output_shapes
:*
seed2ггУ
}
conv1d_1/random_uniform/subSubconv1d_1/random_uniform/maxconv1d_1/random_uniform/min*
_output_shapes
: *
T0

conv1d_1/random_uniform/mulMul%conv1d_1/random_uniform/RandomUniformconv1d_1/random_uniform/sub*"
_output_shapes
:*
T0

conv1d_1/random_uniformAddconv1d_1/random_uniform/mulconv1d_1/random_uniform/min*
T0*"
_output_shapes
:

conv1d_1/kernel
VariableV2*"
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
Ф
conv1d_1/kernel/AssignAssignconv1d_1/kernelconv1d_1/random_uniform*
use_locking(*
T0*"
_class
loc:@conv1d_1/kernel*
validate_shape(*"
_output_shapes
:

conv1d_1/kernel/readIdentityconv1d_1/kernel*
T0*"
_class
loc:@conv1d_1/kernel*"
_output_shapes
:
[
conv1d_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
y
conv1d_1/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
­
conv1d_1/bias/AssignAssignconv1d_1/biasconv1d_1/Const* 
_class
loc:@conv1d_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
t
conv1d_1/bias/readIdentityconv1d_1/bias* 
_class
loc:@conv1d_1/bias*
_output_shapes
:*
T0
l
"conv1d_1/convolution/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
e
#conv1d_1/convolution/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
Љ
conv1d_1/convolution/ExpandDims
ExpandDimsconv1d_1_input#conv1d_1/convolution/ExpandDims/dim*
T0*0
_output_shapes
:џџџџџџџџџЏ*

Tdim0
g
%conv1d_1/convolution/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Љ
!conv1d_1/convolution/ExpandDims_1
ExpandDimsconv1d_1/kernel/read%conv1d_1/convolution/ExpandDims_1/dim*&
_output_shapes
:*

Tdim0*
T0

conv1d_1/convolution/Conv2DConv2Dconv1d_1/convolution/ExpandDims!conv1d_1/convolution/ExpandDims_1*
paddingVALID*0
_output_shapes
:џџџџџџџџџЌ*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

conv1d_1/convolution/SqueezeSqueezeconv1d_1/convolution/Conv2D*,
_output_shapes
:џџџџџџџџџЌ*
squeeze_dims
*
T0
k
conv1d_1/Reshape/shapeConst*
_output_shapes
:*!
valueB"         *
dtype0

conv1d_1/ReshapeReshapeconv1d_1/bias/readconv1d_1/Reshape/shape*
T0*
Tshape0*"
_output_shapes
:
z
conv1d_1/addAddconv1d_1/convolution/Squeezeconv1d_1/Reshape*
T0*,
_output_shapes
:џџџџџџџџџЌ
Z
conv1d_1/ReluReluconv1d_1/add*
T0*,
_output_shapes
:џџџџџџџџџЌ
r
conv1d_2/random_uniform/shapeConst*!
valueB"      @   *
dtype0*
_output_shapes
:
`
conv1d_2/random_uniform/minConst*
valueB
 *ьбН*
dtype0*
_output_shapes
: 
`
conv1d_2/random_uniform/maxConst*
valueB
 *ьб=*
dtype0*
_output_shapes
: 
Ў
%conv1d_2/random_uniform/RandomUniformRandomUniformconv1d_2/random_uniform/shape*
dtype0*"
_output_shapes
:@*
seed2пы*
seedБџх)*
T0
}
conv1d_2/random_uniform/subSubconv1d_2/random_uniform/maxconv1d_2/random_uniform/min*
T0*
_output_shapes
: 

conv1d_2/random_uniform/mulMul%conv1d_2/random_uniform/RandomUniformconv1d_2/random_uniform/sub*"
_output_shapes
:@*
T0

conv1d_2/random_uniformAddconv1d_2/random_uniform/mulconv1d_2/random_uniform/min*"
_output_shapes
:@*
T0

conv1d_2/kernel
VariableV2*
shape:@*
shared_name *
dtype0*"
_output_shapes
:@*
	container 
Ф
conv1d_2/kernel/AssignAssignconv1d_2/kernelconv1d_2/random_uniform*
T0*"
_class
loc:@conv1d_2/kernel*
validate_shape(*"
_output_shapes
:@*
use_locking(

conv1d_2/kernel/readIdentityconv1d_2/kernel*"
_output_shapes
:@*
T0*"
_class
loc:@conv1d_2/kernel
[
conv1d_2/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv1d_2/bias
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
­
conv1d_2/bias/AssignAssignconv1d_2/biasconv1d_2/Const*
use_locking(*
T0* 
_class
loc:@conv1d_2/bias*
validate_shape(*
_output_shapes
:@
t
conv1d_2/bias/readIdentityconv1d_2/bias*
_output_shapes
:@*
T0* 
_class
loc:@conv1d_2/bias
l
"conv1d_2/convolution/dilation_rateConst*
_output_shapes
:*
valueB:*
dtype0
e
#conv1d_2/convolution/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
Ј
conv1d_2/convolution/ExpandDims
ExpandDimsconv1d_1/Relu#conv1d_2/convolution/ExpandDims/dim*

Tdim0*
T0*0
_output_shapes
:џџџџџџџџџЌ
g
%conv1d_2/convolution/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Љ
!conv1d_2/convolution/ExpandDims_1
ExpandDimsconv1d_2/kernel/read%conv1d_2/convolution/ExpandDims_1/dim*

Tdim0*
T0*&
_output_shapes
:@

conv1d_2/convolution/Conv2DConv2Dconv1d_2/convolution/ExpandDims!conv1d_2/convolution/ExpandDims_1*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*0
_output_shapes
:џџџџџџџџџЅ@

conv1d_2/convolution/SqueezeSqueezeconv1d_2/convolution/Conv2D*,
_output_shapes
:џџџџџџџџџЅ@*
squeeze_dims
*
T0
k
conv1d_2/Reshape/shapeConst*!
valueB"      @   *
dtype0*
_output_shapes
:

conv1d_2/ReshapeReshapeconv1d_2/bias/readconv1d_2/Reshape/shape*
Tshape0*"
_output_shapes
:@*
T0
z
conv1d_2/addAddconv1d_2/convolution/Squeezeconv1d_2/Reshape*
T0*,
_output_shapes
:џџџџџџџџџЅ@
Z
conv1d_2/ReluReluconv1d_2/add*
T0*,
_output_shapes
:џџџџџџџџџЅ@
l
lstm_1/random_uniform/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
^
lstm_1/random_uniform/minConst*
valueB
 *:ЭО*
dtype0*
_output_shapes
: 
^
lstm_1/random_uniform/maxConst*
valueB
 *:Э>*
dtype0*
_output_shapes
: 
І
#lstm_1/random_uniform/RandomUniformRandomUniformlstm_1/random_uniform/shape*
T0*
dtype0*
_output_shapes

:@*
seed2шЫ*
seedБџх)
w
lstm_1/random_uniform/subSublstm_1/random_uniform/maxlstm_1/random_uniform/min*
T0*
_output_shapes
: 

lstm_1/random_uniform/mulMul#lstm_1/random_uniform/RandomUniformlstm_1/random_uniform/sub*
_output_shapes

:@*
T0
{
lstm_1/random_uniformAddlstm_1/random_uniform/mullstm_1/random_uniform/min*
_output_shapes

:@*
T0

lstm_1/kernel
VariableV2*
dtype0*
_output_shapes

:@*
	container *
shape
:@*
shared_name 
И
lstm_1/kernel/AssignAssignlstm_1/kernellstm_1/random_uniform* 
_class
loc:@lstm_1/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
x
lstm_1/kernel/readIdentitylstm_1/kernel*
T0* 
_class
loc:@lstm_1/kernel*
_output_shapes

:@
Ж
%lstm_1/recurrent_kernel/initial_valueConst*Y
valuePBN"@Ё!Пv[C>gfмОAЭ>Ѕb>x1ЅО<gОшzлНнЈ:ЮЗОгЌПуE=>XН Оџk/?)Е>*
dtype0*
_output_shapes

:

lstm_1/recurrent_kernel
VariableV2*
_output_shapes

:*
	container *
shape
:*
shared_name *
dtype0
ц
lstm_1/recurrent_kernel/AssignAssignlstm_1/recurrent_kernel%lstm_1/recurrent_kernel/initial_value*
use_locking(*
T0**
_class 
loc:@lstm_1/recurrent_kernel*
validate_shape(*
_output_shapes

:

lstm_1/recurrent_kernel/readIdentitylstm_1/recurrent_kernel*
T0**
_class 
loc:@lstm_1/recurrent_kernel*
_output_shapes

:
Y
lstm_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
[
lstm_1/Const_1Const*
valueB*  ?*
dtype0*
_output_shapes
:
[
lstm_1/Const_2Const*
valueB*    *
dtype0*
_output_shapes
:
T
lstm_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

lstm_1/concatConcatV2lstm_1/Constlstm_1/Const_1lstm_1/Const_2lstm_1/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
w
lstm_1/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
І
lstm_1/bias/AssignAssignlstm_1/biaslstm_1/concat*
use_locking(*
T0*
_class
loc:@lstm_1/bias*
validate_shape(*
_output_shapes
:
n
lstm_1/bias/readIdentitylstm_1/bias*
_output_shapes
:*
T0*
_class
loc:@lstm_1/bias
k
lstm_1/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
m
lstm_1/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
m
lstm_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
Њ
lstm_1/strided_sliceStridedSlicelstm_1/kernel/readlstm_1/strided_slice/stacklstm_1/strided_slice/stack_1lstm_1/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:@
m
lstm_1/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB"       
o
lstm_1/strided_slice_1/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
В
lstm_1/strided_slice_1StridedSlicelstm_1/kernel/readlstm_1/strided_slice_1/stacklstm_1/strided_slice_1/stack_1lstm_1/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:@
m
lstm_1/strided_slice_2/stackConst*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_2/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
В
lstm_1/strided_slice_2StridedSlicelstm_1/kernel/readlstm_1/strided_slice_2/stacklstm_1/strided_slice_2/stack_1lstm_1/strided_slice_2/stack_2*
_output_shapes

:@*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
m
lstm_1/strided_slice_3/stackConst*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_3/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
В
lstm_1/strided_slice_3StridedSlicelstm_1/kernel/readlstm_1/strided_slice_3/stacklstm_1/strided_slice_3/stack_1lstm_1/strided_slice_3/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:@
m
lstm_1/strided_slice_4/stackConst*
valueB"        *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_4/stack_1Const*
_output_shapes
:*
valueB"       *
dtype0
o
lstm_1/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
М
lstm_1/strided_slice_4StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_4/stacklstm_1/strided_slice_4/stack_1lstm_1/strided_slice_4/stack_2*
_output_shapes

:*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
m
lstm_1/strided_slice_5/stackConst*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_5/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_5/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
М
lstm_1/strided_slice_5StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_5/stacklstm_1/strided_slice_5/stack_1lstm_1/strided_slice_5/stack_2*
new_axis_mask *
end_mask*
_output_shapes

:*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask
m
lstm_1/strided_slice_6/stackConst*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_6/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
М
lstm_1/strided_slice_6StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_6/stacklstm_1/strided_slice_6/stack_1lstm_1/strided_slice_6/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:*
Index0*
T0
m
lstm_1/strided_slice_7/stackConst*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_7/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        
o
lstm_1/strided_slice_7/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
М
lstm_1/strided_slice_7StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_7/stacklstm_1/strided_slice_7/stack_1lstm_1/strided_slice_7/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:
f
lstm_1/strided_slice_8/stackConst*
valueB: *
dtype0*
_output_shapes
:
h
lstm_1/strided_slice_8/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
h
lstm_1/strided_slice_8/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Ќ
lstm_1/strided_slice_8StridedSlicelstm_1/bias/readlstm_1/strided_slice_8/stacklstm_1/strided_slice_8/stack_1lstm_1/strided_slice_8/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0
f
lstm_1/strided_slice_9/stackConst*
valueB:*
dtype0*
_output_shapes
:
h
lstm_1/strided_slice_9/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
h
lstm_1/strided_slice_9/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ќ
lstm_1/strided_slice_9StridedSlicelstm_1/bias/readlstm_1/strided_slice_9/stacklstm_1/strided_slice_9/stack_1lstm_1/strided_slice_9/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0
g
lstm_1/strided_slice_10/stackConst*
valueB:*
dtype0*
_output_shapes
:
i
lstm_1/strided_slice_10/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
i
lstm_1/strided_slice_10/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
А
lstm_1/strided_slice_10StridedSlicelstm_1/bias/readlstm_1/strided_slice_10/stacklstm_1/strided_slice_10/stack_1lstm_1/strided_slice_10/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
:
g
lstm_1/strided_slice_11/stackConst*
dtype0*
_output_shapes
:*
valueB:
i
lstm_1/strided_slice_11/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
i
lstm_1/strided_slice_11/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
А
lstm_1/strided_slice_11StridedSlicelstm_1/bias/readlstm_1/strided_slice_11/stacklstm_1/strided_slice_11/stack_1lstm_1/strided_slice_11/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
T0*
Index0
d
lstm_1/zeros_like	ZerosLikeconv1d_2/Relu*,
_output_shapes
:џџџџџџџџџЅ@*
T0
m
lstm_1/Sum/reduction_indicesConst*
_output_shapes
:*
valueB"      *
dtype0


lstm_1/SumSumlstm_1/zeros_likelstm_1/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:џџџџџџџџџ
`
lstm_1/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

lstm_1/ExpandDims
ExpandDims
lstm_1/Sumlstm_1/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:џџџџџџџџџ
f
lstm_1/Tile/multiplesConst*
valueB"      *
dtype0*
_output_shapes
:

lstm_1/TileTilelstm_1/ExpandDimslstm_1/Tile/multiples*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
h
lstm_1/Tile_1/multiplesConst*
valueB"      *
dtype0*
_output_shapes
:

lstm_1/Tile_1Tilelstm_1/ExpandDimslstm_1/Tile_1/multiples*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
j
lstm_1/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:

lstm_1/transpose	Transposeconv1d_2/Relulstm_1/transpose/perm*
T0*,
_output_shapes
:Ѕџџџџџџџџџ@*
Tperm0
\
lstm_1/ShapeShapelstm_1/transpose*
T0*
out_type0*
_output_shapes
:
g
lstm_1/strided_slice_12/stackConst*
_output_shapes
:*
valueB: *
dtype0
i
lstm_1/strided_slice_12/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
i
lstm_1/strided_slice_12/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Ј
lstm_1/strided_slice_12StridedSlicelstm_1/Shapelstm_1/strided_slice_12/stacklstm_1/strided_slice_12/stack_1lstm_1/strided_slice_12/stack_2*
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask 
g
lstm_1/strided_slice_13/stackConst*
valueB: *
dtype0*
_output_shapes
:
i
lstm_1/strided_slice_13/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
i
lstm_1/strided_slice_13/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Н
lstm_1/strided_slice_13StridedSlicelstm_1/transposelstm_1/strided_slice_13/stacklstm_1/strided_slice_13/stack_1lstm_1/strided_slice_13/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *'
_output_shapes
:џџџџџџџџџ@*
T0*
Index0

lstm_1/MatMulMatMullstm_1/strided_slice_13lstm_1/strided_slice*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ђ
lstm_1/MatMul_1MatMullstm_1/strided_slice_13lstm_1/strided_slice_1*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ђ
lstm_1/MatMul_2MatMullstm_1/strided_slice_13lstm_1/strided_slice_2*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 
Ђ
lstm_1/MatMul_3MatMullstm_1/strided_slice_13lstm_1/strided_slice_3*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

lstm_1/BiasAddBiasAddlstm_1/MatMullstm_1/strided_slice_8*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0

lstm_1/BiasAdd_1BiasAddlstm_1/MatMul_1lstm_1/strided_slice_9*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ

lstm_1/BiasAdd_2BiasAddlstm_1/MatMul_2lstm_1/strided_slice_10*'
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC

lstm_1/BiasAdd_3BiasAddlstm_1/MatMul_3lstm_1/strided_slice_11*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMul_4MatMullstm_1/Tilelstm_1/strided_slice_4*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
d

lstm_1/addAddlstm_1/BiasAddlstm_1/MatMul_4*'
_output_shapes
:џџџџџџџџџ*
T0
Q
lstm_1/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬL>
]

lstm_1/mulMullstm_1/mul/x
lstm_1/add*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/add_1/yConst*
_output_shapes
: *
valueB
 *   ?*
dtype0
a
lstm_1/add_1Add
lstm_1/mullstm_1/add_1/y*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/Const_3Const*
valueB
 *    *
dtype0*
_output_shapes
: 
S
lstm_1/Const_4Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
w
lstm_1/clip_by_value/MinimumMinimumlstm_1/add_1lstm_1/Const_4*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/clip_by_valueMaximumlstm_1/clip_by_value/Minimumlstm_1/Const_3*'
_output_shapes
:џџџџџџџџџ*
T0

lstm_1/MatMul_5MatMullstm_1/Tilelstm_1/strided_slice_5*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
h
lstm_1/add_2Addlstm_1/BiasAdd_1lstm_1/MatMul_5*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/mul_1/xConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
c
lstm_1/mul_1Mullstm_1/mul_1/xlstm_1/add_2*'
_output_shapes
:џџџџџџџџџ*
T0
S
lstm_1/add_3/yConst*
_output_shapes
: *
valueB
 *   ?*
dtype0
c
lstm_1/add_3Addlstm_1/mul_1lstm_1/add_3/y*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/Const_5Const*
valueB
 *    *
dtype0*
_output_shapes
: 
S
lstm_1/Const_6Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
y
lstm_1/clip_by_value_1/MinimumMinimumlstm_1/add_3lstm_1/Const_6*'
_output_shapes
:џџџџџџџџџ*
T0

lstm_1/clip_by_value_1Maximumlstm_1/clip_by_value_1/Minimumlstm_1/Const_5*'
_output_shapes
:џџџџџџџџџ*
T0
l
lstm_1/mul_2Mullstm_1/clip_by_value_1lstm_1/Tile_1*'
_output_shapes
:џџџџџџџџџ*
T0

lstm_1/MatMul_6MatMullstm_1/Tilelstm_1/strided_slice_6*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 
h
lstm_1/add_4Addlstm_1/BiasAdd_2lstm_1/MatMul_6*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/TanhTanhlstm_1/add_4*'
_output_shapes
:џџџџџџџџџ*
T0
h
lstm_1/mul_3Mullstm_1/clip_by_valuelstm_1/Tanh*
T0*'
_output_shapes
:џџџџџџџџџ
a
lstm_1/add_5Addlstm_1/mul_2lstm_1/mul_3*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMul_7MatMullstm_1/Tilelstm_1/strided_slice_7*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
h
lstm_1/add_6Addlstm_1/BiasAdd_3lstm_1/MatMul_7*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/mul_4/xConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
c
lstm_1/mul_4Mullstm_1/mul_4/xlstm_1/add_6*'
_output_shapes
:џџџџџџџџџ*
T0
S
lstm_1/add_7/yConst*
dtype0*
_output_shapes
: *
valueB
 *   ?
c
lstm_1/add_7Addlstm_1/mul_4lstm_1/add_7/y*'
_output_shapes
:џџџџџџџџџ*
T0
S
lstm_1/Const_7Const*
_output_shapes
: *
valueB
 *    *
dtype0
S
lstm_1/Const_8Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
y
lstm_1/clip_by_value_2/MinimumMinimumlstm_1/add_7lstm_1/Const_8*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/clip_by_value_2Maximumlstm_1/clip_by_value_2/Minimumlstm_1/Const_7*
T0*'
_output_shapes
:џџџџџџџџџ
U
lstm_1/Tanh_1Tanhlstm_1/add_5*
T0*'
_output_shapes
:џџџџџџџџџ
l
lstm_1/mul_5Mullstm_1/clip_by_value_2lstm_1/Tanh_1*'
_output_shapes
:џџџџџџџџџ*
T0
ь
lstm_1/TensorArrayTensorArrayV3lstm_1/strided_slice_12*
element_shape:*
clear_after_read(*
dynamic_size( *
identical_element_shapes(* 
tensor_array_name	output_ta*
dtype0*
_output_shapes

:: 
э
lstm_1/TensorArray_1TensorArrayV3lstm_1/strided_slice_12*
tensor_array_name
input_ta*
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
o
lstm_1/TensorArrayUnstack/ShapeShapelstm_1/transpose*
T0*
out_type0*
_output_shapes
:
w
-lstm_1/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
y
/lstm_1/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/lstm_1/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ћ
'lstm_1/TensorArrayUnstack/strided_sliceStridedSlicelstm_1/TensorArrayUnstack/Shape-lstm_1/TensorArrayUnstack/strided_slice/stack/lstm_1/TensorArrayUnstack/strided_slice/stack_1/lstm_1/TensorArrayUnstack/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
g
%lstm_1/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
g
%lstm_1/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
а
lstm_1/TensorArrayUnstack/rangeRange%lstm_1/TensorArrayUnstack/range/start'lstm_1/TensorArrayUnstack/strided_slice%lstm_1/TensorArrayUnstack/range/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0

Alstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3lstm_1/TensorArray_1lstm_1/TensorArrayUnstack/rangelstm_1/transposelstm_1/TensorArray_1:1*
T0*#
_class
loc:@lstm_1/transpose*
_output_shapes
: 
M
lstm_1/timeConst*
value	B : *
dtype0*
_output_shapes
: 
b
lstm_1/while/maximum_iterationsConst*
value
B :Ѕ*
dtype0*
_output_shapes
: 
`
lstm_1/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
Ж
lstm_1/while/EnterEnterlstm_1/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: **

frame_namelstm_1/while/while_context
Ѕ
lstm_1/while/Enter_1Enterlstm_1/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: **

frame_namelstm_1/while/while_context
Ў
lstm_1/while/Enter_2Enterlstm_1/TensorArray:1*
parallel_iterations *
_output_shapes
: **

frame_namelstm_1/while/while_context*
T0*
is_constant( 
Ж
lstm_1/while/Enter_3Enterlstm_1/Tile*
parallel_iterations *'
_output_shapes
:џџџџџџџџџ**

frame_namelstm_1/while/while_context*
T0*
is_constant( 
И
lstm_1/while/Enter_4Enterlstm_1/Tile_1*'
_output_shapes
:џџџџџџџџџ**

frame_namelstm_1/while/while_context*
T0*
is_constant( *
parallel_iterations 
w
lstm_1/while/MergeMergelstm_1/while/Enterlstm_1/while/NextIteration*
T0*
N*
_output_shapes
: : 
}
lstm_1/while/Merge_1Mergelstm_1/while/Enter_1lstm_1/while/NextIteration_1*
N*
_output_shapes
: : *
T0
}
lstm_1/while/Merge_2Mergelstm_1/while/Enter_2lstm_1/while/NextIteration_2*
T0*
N*
_output_shapes
: : 

lstm_1/while/Merge_3Mergelstm_1/while/Enter_3lstm_1/while/NextIteration_3*
T0*
N*)
_output_shapes
:џџџџџџџџџ: 

lstm_1/while/Merge_4Mergelstm_1/while/Enter_4lstm_1/while/NextIteration_4*
T0*
N*)
_output_shapes
:џџџџџџџџџ: 
g
lstm_1/while/LessLesslstm_1/while/Mergelstm_1/while/Less/Enter*
_output_shapes
: *
T0
М
lstm_1/while/Less/EnterEnterlstm_1/while/maximum_iterations*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: **

frame_namelstm_1/while/while_context
m
lstm_1/while/Less_1Lesslstm_1/while/Merge_1lstm_1/while/Less_1/Enter*
_output_shapes
: *
T0
Ж
lstm_1/while/Less_1/EnterEnterlstm_1/strided_slice_12*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: **

frame_namelstm_1/while/while_context
e
lstm_1/while/LogicalAnd
LogicalAndlstm_1/while/Lesslstm_1/while/Less_1*
_output_shapes
: 
R
lstm_1/while/LoopCondLoopCondlstm_1/while/LogicalAnd*
_output_shapes
: 

lstm_1/while/SwitchSwitchlstm_1/while/Mergelstm_1/while/LoopCond*
T0*%
_class
loc:@lstm_1/while/Merge*
_output_shapes
: : 

lstm_1/while/Switch_1Switchlstm_1/while/Merge_1lstm_1/while/LoopCond*
T0*'
_class
loc:@lstm_1/while/Merge_1*
_output_shapes
: : 

lstm_1/while/Switch_2Switchlstm_1/while/Merge_2lstm_1/while/LoopCond*
T0*'
_class
loc:@lstm_1/while/Merge_2*
_output_shapes
: : 
К
lstm_1/while/Switch_3Switchlstm_1/while/Merge_3lstm_1/while/LoopCond*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*
T0*'
_class
loc:@lstm_1/while/Merge_3
К
lstm_1/while/Switch_4Switchlstm_1/while/Merge_4lstm_1/while/LoopCond*
T0*'
_class
loc:@lstm_1/while/Merge_4*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
Y
lstm_1/while/IdentityIdentitylstm_1/while/Switch:1*
T0*
_output_shapes
: 
]
lstm_1/while/Identity_1Identitylstm_1/while/Switch_1:1*
_output_shapes
: *
T0
]
lstm_1/while/Identity_2Identitylstm_1/while/Switch_2:1*
T0*
_output_shapes
: 
n
lstm_1/while/Identity_3Identitylstm_1/while/Switch_3:1*'
_output_shapes
:џџџџџџџџџ*
T0
n
lstm_1/while/Identity_4Identitylstm_1/while/Switch_4:1*
T0*'
_output_shapes
:џџџџџџџџџ
l
lstm_1/while/add/yConst^lstm_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
c
lstm_1/while/addAddlstm_1/while/Identitylstm_1/while/add/y*
_output_shapes
: *
T0
а
lstm_1/while/TensorArrayReadV3TensorArrayReadV3$lstm_1/while/TensorArrayReadV3/Enterlstm_1/while/Identity_1&lstm_1/while/TensorArrayReadV3/Enter_1*'
_output_shapes
:џџџџџџџџџ@*
dtype0
Т
$lstm_1/while/TensorArrayReadV3/EnterEnterlstm_1/TensorArray_1*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0
э
&lstm_1/while/TensorArrayReadV3/Enter_1EnterAlstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: **

frame_namelstm_1/while/while_context
А
lstm_1/while/MatMulMatMullstm_1/while/TensorArrayReadV3lstm_1/while/MatMul/Enter*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
Л
lstm_1/while/MatMul/EnterEnterlstm_1/strided_slice*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:@**

frame_namelstm_1/while/while_context
Д
lstm_1/while/MatMul_1MatMullstm_1/while/TensorArrayReadV3lstm_1/while/MatMul_1/Enter*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
П
lstm_1/while/MatMul_1/EnterEnterlstm_1/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:@**

frame_namelstm_1/while/while_context
Д
lstm_1/while/MatMul_2MatMullstm_1/while/TensorArrayReadV3lstm_1/while/MatMul_2/Enter*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
П
lstm_1/while/MatMul_2/EnterEnterlstm_1/strided_slice_2*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:@**

frame_namelstm_1/while/while_context
Д
lstm_1/while/MatMul_3MatMullstm_1/while/TensorArrayReadV3lstm_1/while/MatMul_3/Enter*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
П
lstm_1/while/MatMul_3/EnterEnterlstm_1/strided_slice_3*
is_constant(*
parallel_iterations *
_output_shapes

:@**

frame_namelstm_1/while/while_context*
T0

lstm_1/while/BiasAddBiasAddlstm_1/while/MatMullstm_1/while/BiasAdd/Enter*'
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
К
lstm_1/while/BiasAdd/EnterEnterlstm_1/strided_slice_8*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*
is_constant(*
parallel_iterations 

lstm_1/while/BiasAdd_1BiasAddlstm_1/while/MatMul_1lstm_1/while/BiasAdd_1/Enter*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
М
lstm_1/while/BiasAdd_1/EnterEnterlstm_1/strided_slice_9*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelstm_1/while/while_context

lstm_1/while/BiasAdd_2BiasAddlstm_1/while/MatMul_2lstm_1/while/BiasAdd_2/Enter*'
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
Н
lstm_1/while/BiasAdd_2/EnterEnterlstm_1/strided_slice_10*
parallel_iterations *
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*
is_constant(

lstm_1/while/BiasAdd_3BiasAddlstm_1/while/MatMul_3lstm_1/while/BiasAdd_3/Enter*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
Н
lstm_1/while/BiasAdd_3/EnterEnterlstm_1/strided_slice_11*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelstm_1/while/while_context
­
lstm_1/while/MatMul_4MatMullstm_1/while/Identity_3lstm_1/while/MatMul_4/Enter*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
П
lstm_1/while/MatMul_4/EnterEnterlstm_1/strided_slice_4*
is_constant(*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context*
T0
x
lstm_1/while/add_1Addlstm_1/while/BiasAddlstm_1/while/MatMul_4*
T0*'
_output_shapes
:џџџџџџџџџ
o
lstm_1/while/mul/xConst^lstm_1/while/Identity*
_output_shapes
: *
valueB
 *ЭЬL>*
dtype0
q
lstm_1/while/mulMullstm_1/while/mul/xlstm_1/while/add_1*'
_output_shapes
:џџџџџџџџџ*
T0
q
lstm_1/while/add_2/yConst^lstm_1/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 
s
lstm_1/while/add_2Addlstm_1/while/mullstm_1/while/add_2/y*
T0*'
_output_shapes
:џџџџџџџџџ
o
lstm_1/while/ConstConst^lstm_1/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
q
lstm_1/while/Const_1Const^lstm_1/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 

"lstm_1/while/clip_by_value/MinimumMinimumlstm_1/while/add_2lstm_1/while/Const_1*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/while/clip_by_valueMaximum"lstm_1/while/clip_by_value/Minimumlstm_1/while/Const*
T0*'
_output_shapes
:џџџџџџџџџ
­
lstm_1/while/MatMul_5MatMullstm_1/while/Identity_3lstm_1/while/MatMul_5/Enter*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
П
lstm_1/while/MatMul_5/EnterEnterlstm_1/strided_slice_5*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context
z
lstm_1/while/add_3Addlstm_1/while/BiasAdd_1lstm_1/while/MatMul_5*'
_output_shapes
:џџџџџџџџџ*
T0
q
lstm_1/while/mul_1/xConst^lstm_1/while/Identity*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
u
lstm_1/while/mul_1Mullstm_1/while/mul_1/xlstm_1/while/add_3*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/add_4/yConst^lstm_1/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 
u
lstm_1/while/add_4Addlstm_1/while/mul_1lstm_1/while/add_4/y*'
_output_shapes
:џџџџџџџџџ*
T0
q
lstm_1/while/Const_2Const^lstm_1/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *    
q
lstm_1/while/Const_3Const^lstm_1/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$lstm_1/while/clip_by_value_1/MinimumMinimumlstm_1/while/add_4lstm_1/while/Const_3*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/while/clip_by_value_1Maximum$lstm_1/while/clip_by_value_1/Minimumlstm_1/while/Const_2*'
_output_shapes
:џџџџџџџџџ*
T0

lstm_1/while/mul_2Mullstm_1/while/clip_by_value_1lstm_1/while/Identity_4*'
_output_shapes
:џџџџџџџџџ*
T0
­
lstm_1/while/MatMul_6MatMullstm_1/while/Identity_3lstm_1/while/MatMul_6/Enter*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
П
lstm_1/while/MatMul_6/EnterEnterlstm_1/strided_slice_6*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context
z
lstm_1/while/add_5Addlstm_1/while/BiasAdd_2lstm_1/while/MatMul_6*
T0*'
_output_shapes
:џџџџџџџџџ
_
lstm_1/while/TanhTanhlstm_1/while/add_5*'
_output_shapes
:џџџџџџџџџ*
T0
z
lstm_1/while/mul_3Mullstm_1/while/clip_by_valuelstm_1/while/Tanh*
T0*'
_output_shapes
:џџџџџџџџџ
s
lstm_1/while/add_6Addlstm_1/while/mul_2lstm_1/while/mul_3*'
_output_shapes
:џџџџџџџџџ*
T0
­
lstm_1/while/MatMul_7MatMullstm_1/while/Identity_3lstm_1/while/MatMul_7/Enter*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
П
lstm_1/while/MatMul_7/EnterEnterlstm_1/strided_slice_7*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context*
T0*
is_constant(
z
lstm_1/while/add_7Addlstm_1/while/BiasAdd_3lstm_1/while/MatMul_7*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/mul_4/xConst^lstm_1/while/Identity*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
u
lstm_1/while/mul_4Mullstm_1/while/mul_4/xlstm_1/while/add_7*'
_output_shapes
:џџџџџџџџџ*
T0
q
lstm_1/while/add_8/yConst^lstm_1/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *   ?
u
lstm_1/while/add_8Addlstm_1/while/mul_4lstm_1/while/add_8/y*'
_output_shapes
:џџџџџџџџџ*
T0
q
lstm_1/while/Const_4Const^lstm_1/while/Identity*
_output_shapes
: *
valueB
 *    *
dtype0
q
lstm_1/while/Const_5Const^lstm_1/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$lstm_1/while/clip_by_value_2/MinimumMinimumlstm_1/while/add_8lstm_1/while/Const_5*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/while/clip_by_value_2Maximum$lstm_1/while/clip_by_value_2/Minimumlstm_1/while/Const_4*
T0*'
_output_shapes
:џџџџџџџџџ
a
lstm_1/while/Tanh_1Tanhlstm_1/while/add_6*
T0*'
_output_shapes
:џџџџџџџџџ
~
lstm_1/while/mul_5Mullstm_1/while/clip_by_value_2lstm_1/while/Tanh_1*'
_output_shapes
:џџџџџџџџџ*
T0

0lstm_1/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV36lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/Enterlstm_1/while/Identity_1lstm_1/while/mul_5lstm_1/while/Identity_2*
_output_shapes
: *
T0*%
_class
loc:@lstm_1/while/mul_5
љ
6lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterlstm_1/TensorArray*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0
n
lstm_1/while/add_9/yConst^lstm_1/while/Identity*
dtype0*
_output_shapes
: *
value	B :
i
lstm_1/while/add_9Addlstm_1/while/Identity_1lstm_1/while/add_9/y*
_output_shapes
: *
T0
^
lstm_1/while/NextIterationNextIterationlstm_1/while/add*
T0*
_output_shapes
: 
b
lstm_1/while/NextIteration_1NextIterationlstm_1/while/add_9*
_output_shapes
: *
T0

lstm_1/while/NextIteration_2NextIteration0lstm_1/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
s
lstm_1/while/NextIteration_3NextIterationlstm_1/while/mul_5*'
_output_shapes
:џџџџџџџџџ*
T0
s
lstm_1/while/NextIteration_4NextIterationlstm_1/while/add_6*
T0*'
_output_shapes
:џџџџџџџџџ
O
lstm_1/while/ExitExitlstm_1/while/Switch*
T0*
_output_shapes
: 
S
lstm_1/while/Exit_1Exitlstm_1/while/Switch_1*
T0*
_output_shapes
: 
S
lstm_1/while/Exit_2Exitlstm_1/while/Switch_2*
T0*
_output_shapes
: 
d
lstm_1/while/Exit_3Exitlstm_1/while/Switch_3*'
_output_shapes
:џџџџџџџџџ*
T0
d
lstm_1/while/Exit_4Exitlstm_1/while/Switch_4*
T0*'
_output_shapes
:џџџџџџџџџ
І
)lstm_1/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3lstm_1/TensorArraylstm_1/while/Exit_2*%
_class
loc:@lstm_1/TensorArray*
_output_shapes
: 

#lstm_1/TensorArrayStack/range/startConst*%
_class
loc:@lstm_1/TensorArray*
value	B : *
dtype0*
_output_shapes
: 

#lstm_1/TensorArrayStack/range/deltaConst*%
_class
loc:@lstm_1/TensorArray*
value	B :*
dtype0*
_output_shapes
: 
ѓ
lstm_1/TensorArrayStack/rangeRange#lstm_1/TensorArrayStack/range/start)lstm_1/TensorArrayStack/TensorArraySizeV3#lstm_1/TensorArrayStack/range/delta*

Tidx0*%
_class
loc:@lstm_1/TensorArray*#
_output_shapes
:џџџџџџџџџ

+lstm_1/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3lstm_1/TensorArraylstm_1/TensorArrayStack/rangelstm_1/while/Exit_2*$
element_shape:џџџџџџџџџ*%
_class
loc:@lstm_1/TensorArray*
dtype0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
N
lstm_1/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
U

lstm_1/subSublstm_1/while/Exit_1lstm_1/sub/y*
T0*
_output_shapes
: 

lstm_1/TensorArrayReadV3TensorArrayReadV3lstm_1/TensorArray
lstm_1/sublstm_1/while/Exit_2*
dtype0*'
_output_shapes
:џџџџџџџџџ
l
lstm_1/transpose_1/permConst*
_output_shapes
:*!
valueB"          *
dtype0
Б
lstm_1/transpose_1	Transpose+lstm_1/TensorArrayStack/TensorArrayGatherV3lstm_1/transpose_1/perm*
Tperm0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
f
$dropout_1/keras_learning_phase/inputConst*
_output_shapes
: *
value	B
 Z *
dtype0


dropout_1/keras_learning_phasePlaceholderWithDefault$dropout_1/keras_learning_phase/input*
dtype0
*
_output_shapes
: *
shape: 

dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes
: : 
]
dropout_1/cond/switch_tIdentitydropout_1/cond/Switch:1*
_output_shapes
: *
T0

[
dropout_1/cond/switch_fIdentitydropout_1/cond/Switch*
T0
*
_output_shapes
: 
c
dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
: *
T0

s
dropout_1/cond/mul/yConst^dropout_1/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 

dropout_1/cond/mulMuldropout_1/cond/mul/Switch:1dropout_1/cond/mul/y*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0
е
dropout_1/cond/mul/SwitchSwitchlstm_1/transpose_1dropout_1/cond/pred_id*T
_output_shapesB
@:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ*
T0*%
_class
loc:@lstm_1/transpose_1

 dropout_1/cond/dropout/keep_probConst^dropout_1/cond/switch_t*
valueB
 *fff?*
dtype0*
_output_shapes
: 
n
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
T0*
out_type0*
_output_shapes
:

)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
Ь
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
seed2ФГp*
seedБџх)*
T0*
dtype0
Ї
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Я
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
С
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
Љ
dropout_1/cond/dropout/addAdd dropout_1/cond/dropout/keep_prob%dropout_1/cond/dropout/random_uniform*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0

dropout_1/cond/dropout/FloorFloordropout_1/cond/dropout/add*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

dropout_1/cond/dropout/divRealDivdropout_1/cond/mul dropout_1/cond/dropout/keep_prob*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/divdropout_1/cond/dropout/Floor*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
г
dropout_1/cond/Switch_1Switchlstm_1/transpose_1dropout_1/cond/pred_id*
T0*%
_class
loc:@lstm_1/transpose_1*T
_output_shapesB
@:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ

dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*
N*6
_output_shapes$
":џџџџџџџџџџџџџџџџџџ: *
T0
c
flatten_1/ShapeShapedropout_1/cond/Merge*
out_type0*
_output_shapes
:*
T0
g
flatten_1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Џ
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:
Y
flatten_1/ConstConst*
_output_shapes
:*
valueB: *
dtype0
~
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
\
flatten_1/stack/0Const*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
t
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*
T0*

axis *
N*
_output_shapes
:

flatten_1/ReshapeReshapedropout_1/cond/Mergeflatten_1/stack*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0*
Tshape0
m
dense_1/random_uniform/shapeConst*
valueB"J     *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
_output_shapes
: *
valueB
 *hъН*
dtype0
_
dense_1/random_uniform/maxConst*
valueB
 *hъ=*
dtype0*
_output_shapes
: 
Њ
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
T0*
dtype0* 
_output_shapes
:
Ъ*
seed2ЁЅХ*
seedБџх)
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 

dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub* 
_output_shapes
:
Ъ*
T0

dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min* 
_output_shapes
:
Ъ*
T0

dense_1/kernel
VariableV2*
dtype0* 
_output_shapes
:
Ъ*
	container *
shape:
Ъ*
shared_name 
О
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(* 
_output_shapes
:
Ъ
}
dense_1/kernel/readIdentitydense_1/kernel* 
_output_shapes
:
Ъ*
T0*!
_class
loc:@dense_1/kernel
\
dense_1/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
z
dense_1/bias
VariableV2*
_output_shapes	
:*
	container *
shape:*
shared_name *
dtype0
Њ
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(
r
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes	
:

dense_1/MatMulMatMulflatten_1/Reshapedense_1/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
X
dense_1/ReluReludense_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes
: : *
T0

]
dropout_2/cond/switch_tIdentitydropout_2/cond/Switch:1*
T0
*
_output_shapes
: 
[
dropout_2/cond/switch_fIdentitydropout_2/cond/Switch*
_output_shapes
: *
T0

c
dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
: *
T0

s
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 

dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*
T0*(
_output_shapes
:џџџџџџџџџ
Б
dropout_2/cond/mul/SwitchSwitchdense_1/Reludropout_2/cond/pred_id*
_class
loc:@dense_1/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*
T0

 dropout_2/cond/dropout/keep_probConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *   ?
n
dropout_2/cond/dropout/ShapeShapedropout_2/cond/mul*
_output_shapes
:*
T0*
out_type0

)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
_output_shapes
: *
valueB
 *    *
dtype0

)dropout_2/cond/dropout/random_uniform/maxConst^dropout_2/cond/switch_t*
_output_shapes
: *
valueB
 *  ?*
dtype0
Р
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
T0*
dtype0*(
_output_shapes
:џџџџџџџџџ*
seed2,*
seedБџх)
Ї
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
У
)dropout_2/cond/dropout/random_uniform/mulMul3dropout_2/cond/dropout/random_uniform/RandomUniform)dropout_2/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:џџџџџџџџџ
Е
%dropout_2/cond/dropout/random_uniformAdd)dropout_2/cond/dropout/random_uniform/mul)dropout_2/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_2/cond/dropout/addAdd dropout_2/cond/dropout/keep_prob%dropout_2/cond/dropout/random_uniform*
T0*(
_output_shapes
:џџџџџџџџџ
t
dropout_2/cond/dropout/FloorFloordropout_2/cond/dropout/add*(
_output_shapes
:џџџџџџџџџ*
T0

dropout_2/cond/dropout/divRealDivdropout_2/cond/mul dropout_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_2/cond/dropout/mulMuldropout_2/cond/dropout/divdropout_2/cond/dropout/Floor*(
_output_shapes
:џџџџџџџџџ*
T0
Џ
dropout_2/cond/Switch_1Switchdense_1/Reludropout_2/cond/pred_id*
T0*
_class
loc:@dense_1/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ

dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul*
N**
_output_shapes
:џџџџџџџџџ: *
T0
m
dense_2/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
_
dense_2/random_uniform/minConst*
valueB
 *qФО*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
_output_shapes
: *
valueB
 *qФ>*
dtype0
Њ
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
seedБџх)*
T0*
dtype0* 
_output_shapes
:
*
seed2ЛЅЎ
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 

dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0* 
_output_shapes
:


dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0* 
_output_shapes
:


dense_2/kernel
VariableV2*
shared_name *
dtype0* 
_output_shapes
:
*
	container *
shape:

О
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*!
_class
loc:@dense_2/kernel
}
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel* 
_output_shapes
:

\
dense_2/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
z
dense_2/bias
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Њ
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes	
:
r
dense_2/bias/readIdentitydense_2/bias*
_class
loc:@dense_2/bias*
_output_shapes	
:*
T0

dense_2/MatMulMatMuldropout_2/cond/Mergedense_2/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
X
dense_2/ReluReludense_2/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_3/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes
: : *
T0

]
dropout_3/cond/switch_tIdentitydropout_3/cond/Switch:1*
T0
*
_output_shapes
: 
[
dropout_3/cond/switch_fIdentitydropout_3/cond/Switch*
T0
*
_output_shapes
: 
c
dropout_3/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
: 
s
dropout_3/cond/mul/yConst^dropout_3/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?

dropout_3/cond/mulMuldropout_3/cond/mul/Switch:1dropout_3/cond/mul/y*
T0*(
_output_shapes
:џџџџџџџџџ
Б
dropout_3/cond/mul/SwitchSwitchdense_2/Reludropout_3/cond/pred_id*
T0*
_class
loc:@dense_2/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ

 dropout_3/cond/dropout/keep_probConst^dropout_3/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
n
dropout_3/cond/dropout/ShapeShapedropout_3/cond/mul*
T0*
out_type0*
_output_shapes
:

)dropout_3/cond/dropout/random_uniform/minConst^dropout_3/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    

)dropout_3/cond/dropout/random_uniform/maxConst^dropout_3/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
С
3dropout_3/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_3/cond/dropout/Shape*
T0*
dtype0*(
_output_shapes
:џџџџџџџџџ*
seed2зЌй*
seedБџх)
Ї
)dropout_3/cond/dropout/random_uniform/subSub)dropout_3/cond/dropout/random_uniform/max)dropout_3/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
У
)dropout_3/cond/dropout/random_uniform/mulMul3dropout_3/cond/dropout/random_uniform/RandomUniform)dropout_3/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:џџџџџџџџџ
Е
%dropout_3/cond/dropout/random_uniformAdd)dropout_3/cond/dropout/random_uniform/mul)dropout_3/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_3/cond/dropout/addAdd dropout_3/cond/dropout/keep_prob%dropout_3/cond/dropout/random_uniform*(
_output_shapes
:џџџџџџџџџ*
T0
t
dropout_3/cond/dropout/FloorFloordropout_3/cond/dropout/add*(
_output_shapes
:џџџџџџџџџ*
T0

dropout_3/cond/dropout/divRealDivdropout_3/cond/mul dropout_3/cond/dropout/keep_prob*(
_output_shapes
:џџџџџџџџџ*
T0

dropout_3/cond/dropout/mulMuldropout_3/cond/dropout/divdropout_3/cond/dropout/Floor*(
_output_shapes
:џџџџџџџџџ*
T0
Џ
dropout_3/cond/Switch_1Switchdense_2/Reludropout_3/cond/pred_id*
T0*
_class
loc:@dense_2/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ

dropout_3/cond/MergeMergedropout_3/cond/Switch_1dropout_3/cond/dropout/mul*
N**
_output_shapes
:џџџџџџџџџ: *
T0
m
dense_3/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
_
dense_3/random_uniform/minConst*
valueB
 *ЃЎXО*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
valueB
 *ЃЎX>*
dtype0*
_output_shapes
: 
Ј
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
dtype0*
_output_shapes
:	*
seed2Љ*
seedБџх)*
T0
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
T0*
_output_shapes
: 

dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
_output_shapes
:	*
T0

dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
T0*
_output_shapes
:	

dense_3/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes
:	*
	container *
shape:	
Н
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
_output_shapes
:	*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(
|
dense_3/kernel/readIdentitydense_3/kernel*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes
:	
Z
dense_3/ConstConst*
_output_shapes
:*
valueB*    *
dtype0
x
dense_3/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Љ
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:
q
dense_3/bias/readIdentitydense_3/bias*
T0*
_class
loc:@dense_3/bias*
_output_shapes
:

dense_3/MatMulMatMuldropout_3/cond/Mergedense_3/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
]
dense_3/SoftmaxSoftmaxdense_3/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
_
Adam/iterations/initial_valueConst*
dtype0	*
_output_shapes
: *
value	B	 R 
s
Adam/iterations
VariableV2*
shared_name *
dtype0	*
_output_shapes
: *
	container *
shape: 
О
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*
use_locking(*
T0	*"
_class
loc:@Adam/iterations*
validate_shape(*
_output_shapes
: 
v
Adam/iterations/readIdentityAdam/iterations*
T0	*"
_class
loc:@Adam/iterations*
_output_shapes
: 
Z
Adam/lr/initial_valueConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
k
Adam/lr
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 

Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
_class
loc:@Adam/lr*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
^
Adam/lr/readIdentityAdam/lr*
T0*
_class
loc:@Adam/lr*
_output_shapes
: 
^
Adam/beta_1/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
o
Adam/beta_1
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Ў
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
use_locking(*
T0*
_class
loc:@Adam/beta_1*
validate_shape(*
_output_shapes
: 
j
Adam/beta_1/readIdentityAdam/beta_1*
T0*
_class
loc:@Adam/beta_1*
_output_shapes
: 
^
Adam/beta_2/initial_valueConst*
valueB
 *wО?*
dtype0*
_output_shapes
: 
o
Adam/beta_2
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Ў
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/beta_2*
validate_shape(
j
Adam/beta_2/readIdentityAdam/beta_2*
T0*
_class
loc:@Adam/beta_2*
_output_shapes
: 
]
Adam/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n

Adam/decay
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Њ
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
use_locking(*
T0*
_class
loc:@Adam/decay*
validate_shape(*
_output_shapes
: 
g
Adam/decay/readIdentity
Adam/decay*
_output_shapes
: *
T0*
_class
loc:@Adam/decay

dense_3_targetPlaceholder*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ*
dtype0
q
dense_3_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
r
'loss/dense_3_loss/Sum/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ѕ
loss/dense_3_loss/SumSumdense_3/Softmax'loss/dense_3_loss/Sum/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџ
~
loss/dense_3_loss/truedivRealDivdense_3/Softmaxloss/dense_3_loss/Sum*
T0*'
_output_shapes
:џџџџџџџџџ
\
loss/dense_3_loss/ConstConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
\
loss/dense_3_loss/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
loss/dense_3_loss/subSubloss/dense_3_loss/sub/xloss/dense_3_loss/Const*
_output_shapes
: *
T0

'loss/dense_3_loss/clip_by_value/MinimumMinimumloss/dense_3_loss/truedivloss/dense_3_loss/sub*
T0*'
_output_shapes
:џџџџџџџџџ

loss/dense_3_loss/clip_by_valueMaximum'loss/dense_3_loss/clip_by_value/Minimumloss/dense_3_loss/Const*'
_output_shapes
:џџџџџџџџџ*
T0
o
loss/dense_3_loss/LogLogloss/dense_3_loss/clip_by_value*'
_output_shapes
:џџџџџџџџџ*
T0
u
loss/dense_3_loss/mulMuldense_3_targetloss/dense_3_loss/Log*'
_output_shapes
:џџџџџџџџџ*
T0
t
)loss/dense_3_loss/Sum_1/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ћ
loss/dense_3_loss/Sum_1Sumloss/dense_3_loss/mul)loss/dense_3_loss/Sum_1/reduction_indices*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( *

Tidx0*
T0
c
loss/dense_3_loss/NegNegloss/dense_3_loss/Sum_1*
T0*#
_output_shapes
:џџџџџџџџџ
k
(loss/dense_3_loss/Mean/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
Њ
loss/dense_3_loss/MeanMeanloss/dense_3_loss/Neg(loss/dense_3_loss/Mean/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( *

Tidx0
|
loss/dense_3_loss/mul_1Mulloss/dense_3_loss/Meandense_3_sample_weights*
T0*#
_output_shapes
:џџџџџџџџџ
a
loss/dense_3_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss/dense_3_loss/NotEqualNotEqualdense_3_sample_weightsloss/dense_3_loss/NotEqual/y*#
_output_shapes
:џџџџџџџџџ*
T0

loss/dense_3_loss/CastCastloss/dense_3_loss/NotEqual*

SrcT0
*
Truncate( *#
_output_shapes
:џџџџџџџџџ*

DstT0
c
loss/dense_3_loss/Const_1Const*
_output_shapes
:*
valueB: *
dtype0

loss/dense_3_loss/Mean_1Meanloss/dense_3_loss/Castloss/dense_3_loss/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 

loss/dense_3_loss/truediv_1RealDivloss/dense_3_loss/mul_1loss/dense_3_loss/Mean_1*
T0*#
_output_shapes
:џџџџџџџџџ
c
loss/dense_3_loss/Const_2Const*
valueB: *
dtype0*
_output_shapes
:

loss/dense_3_loss/Mean_2Meanloss/dense_3_loss/truediv_1loss/dense_3_loss/Const_2*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
V
loss/mulMul
loss/mul/xloss/dense_3_loss/Mean_2*
T0*
_output_shapes
: 
g
metrics/acc/ArgMax/dimensionConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

metrics/acc/ArgMaxArgMaxdense_3_targetmetrics/acc/ArgMax/dimension*
output_type0	*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
i
metrics/acc/ArgMax_1/dimensionConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

metrics/acc/ArgMax_1ArgMaxdense_3/Softmaxmetrics/acc/ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:џџџџџџџџџ*

Tidx0
r
metrics/acc/EqualEqualmetrics/acc/ArgMaxmetrics/acc/ArgMax_1*#
_output_shapes
:џџџџџџџџџ*
T0	
x
metrics/acc/CastCastmetrics/acc/Equal*

SrcT0
*
Truncate( *#
_output_shapes
:џџџџџџџџџ*

DstT0
[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
{
metrics/acc/MeanMeanmetrics/acc/Castmetrics/acc/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
}
training/Adam/gradients/ShapeConst*
_class
loc:@loss/mul*
valueB *
dtype0*
_output_shapes
: 

!training/Adam/gradients/grad_ys_0Const*
_output_shapes
: *
_class
loc:@loss/mul*
valueB
 *  ?*
dtype0
Ж
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
T0*
_class
loc:@loss/mul*

index_type0*
_output_shapes
: 

training/Adam/gradients/f_countConst*&
_class
loc:@lstm_1/while/Exit_2*
value	B : *
dtype0*
_output_shapes
: 
ю
!training/Adam/gradients/f_count_1Entertraining/Adam/gradients/f_count*
T0*&
_class
loc:@lstm_1/while/Exit_2*
parallel_iterations *
is_constant( *
_output_shapes
: **

frame_namelstm_1/while/while_context
Ф
training/Adam/gradients/MergeMerge!training/Adam/gradients/f_count_1%training/Adam/gradients/NextIteration*
_output_shapes
: : *
T0*&
_class
loc:@lstm_1/while/Exit_2*
N
Љ
training/Adam/gradients/SwitchSwitchtraining/Adam/gradients/Mergelstm_1/while/LoopCond*
T0*&
_class
loc:@lstm_1/while/Exit_2*
_output_shapes
: : 

training/Adam/gradients/Add/yConst^lstm_1/while/Identity*
dtype0*
_output_shapes
: *&
_class
loc:@lstm_1/while/Exit_2*
value	B :
Ќ
training/Adam/gradients/AddAdd training/Adam/gradients/Switch:1training/Adam/gradients/Add/y*
T0*&
_class
loc:@lstm_1/while/Exit_2*
_output_shapes
: 

%training/Adam/gradients/NextIterationNextIterationtraining/Adam/gradients/AddH^training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPushV2H^training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPushV2l^training/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2R^training/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPushV2_1R^training/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPushV2R^training/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPushV2_1R^training/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPushV2R^training/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPushV2_1R^training/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPushV2_1R^training/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPushV2_1R^training/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPushV2b^training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPushV2V^training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPushV2d^training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPushV2X^training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPushV2\^training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPushV2S^training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPushV2d^training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPushV2X^training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPushV2\^training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPushV2S^training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPushV2Z^training/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPushV2Q^training/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPushV2R^training/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPushV2@^training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPushV2R^training/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPushV2_1@^training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPushV2B^training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPushV2R^training/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPushV2_1@^training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPushV2B^training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPushV2R^training/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPushV2@^training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPushV2R^training/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPushV2_1@^training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPushV2B^training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPushV2P^training/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPushV2>^training/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPushV2*&
_class
loc:@lstm_1/while/Exit_2*
_output_shapes
: *
T0

!training/Adam/gradients/f_count_2Exittraining/Adam/gradients/Switch*
T0*&
_class
loc:@lstm_1/while/Exit_2*
_output_shapes
: 

training/Adam/gradients/b_countConst*&
_class
loc:@lstm_1/while/Exit_2*
value	B :*
dtype0*
_output_shapes
: 

!training/Adam/gradients/b_count_1Enter!training/Adam/gradients/f_count_2*
_output_shapes
: *B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*&
_class
loc:@lstm_1/while/Exit_2*
parallel_iterations *
is_constant( 
Ш
training/Adam/gradients/Merge_1Merge!training/Adam/gradients/b_count_1'training/Adam/gradients/NextIteration_1*
T0*&
_class
loc:@lstm_1/while/Exit_2*
N*
_output_shapes
: : 
Ъ
$training/Adam/gradients/GreaterEqualGreaterEqualtraining/Adam/gradients/Merge_1*training/Adam/gradients/GreaterEqual/Enter*&
_class
loc:@lstm_1/while/Exit_2*
_output_shapes
: *
T0

*training/Adam/gradients/GreaterEqual/EnterEntertraining/Adam/gradients/b_count*
is_constant(*
_output_shapes
: *B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*&
_class
loc:@lstm_1/while/Exit_2*
parallel_iterations 

!training/Adam/gradients/b_count_2LoopCond$training/Adam/gradients/GreaterEqual*&
_class
loc:@lstm_1/while/Exit_2*
_output_shapes
: 
Й
 training/Adam/gradients/Switch_1Switchtraining/Adam/gradients/Merge_1!training/Adam/gradients/b_count_2*
_output_shapes
: : *
T0*&
_class
loc:@lstm_1/while/Exit_2
Л
training/Adam/gradients/SubSub"training/Adam/gradients/Switch_1:1*training/Adam/gradients/GreaterEqual/Enter*
_output_shapes
: *
T0*&
_class
loc:@lstm_1/while/Exit_2

'training/Adam/gradients/NextIteration_1NextIterationtraining/Adam/gradients/Subg^training/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0*&
_class
loc:@lstm_1/while/Exit_2*
_output_shapes
: 

!training/Adam/gradients/b_count_3Exit training/Adam/gradients/Switch_1*
_output_shapes
: *
T0*&
_class
loc:@lstm_1/while/Exit_2
І
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/dense_3_loss/Mean_2*
_output_shapes
: *
T0*
_class
loc:@loss/mul

+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
К
Ctraining/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Reshape/shapeConst*
_output_shapes
:*+
_class!
loc:@loss/dense_3_loss/Mean_2*
valueB:*
dtype0

=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Ctraining/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Reshape/shape*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*
Tshape0
У
;training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/ShapeShapeloss/dense_3_loss/truediv_1*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*
out_type0
Ћ
:training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/TileTile=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Reshape;training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Shape*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*#
_output_shapes
:џџџџџџџџџ*

Tmultiples0
Х
=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Shape_1Shapeloss/dense_3_loss/truediv_1*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*
out_type0
­
=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Shape_2Const*+
_class!
loc:@loss/dense_3_loss/Mean_2*
valueB *
dtype0*
_output_shapes
: 
В
;training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/ConstConst*
dtype0*
_output_shapes
:*+
_class!
loc:@loss/dense_3_loss/Mean_2*
valueB: 
Љ
:training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/ProdProd=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Shape_1;training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Const*
	keep_dims( *

Tidx0*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*
_output_shapes
: 
Д
=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Const_1Const*+
_class!
loc:@loss/dense_3_loss/Mean_2*
valueB: *
dtype0*
_output_shapes
:
­
<training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Prod_1Prod=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Shape_2=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Const_1*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*
_output_shapes
: *
	keep_dims( *

Tidx0
Ў
?training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Maximum/yConst*+
_class!
loc:@loss/dense_3_loss/Mean_2*
value	B :*
dtype0*
_output_shapes
: 

=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/MaximumMaximum<training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Prod_1?training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Maximum/y*+
_class!
loc:@loss/dense_3_loss/Mean_2*
_output_shapes
: *
T0

>training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/floordivFloorDiv:training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Prod=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Maximum*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*
_output_shapes
: 
я
:training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/CastCast>training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0*+
_class!
loc:@loss/dense_3_loss/Mean_2*
Truncate( 

=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/truedivRealDiv:training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Tile:training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Cast*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*#
_output_shapes
:џџџџџџџџџ
Х
>training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/ShapeShapeloss/dense_3_loss/mul_1*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
out_type0*
_output_shapes
:
Г
@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Shape_1Const*
dtype0*
_output_shapes
: *.
_class$
" loc:@loss/dense_3_loss/truediv_1*
valueB 
ж
Ntraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgs>training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Shape@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1

@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDivRealDiv=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/truedivloss/dense_3_loss/Mean_1*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ
Х
<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/SumSum@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDivNtraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1
Е
@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/ReshapeReshape<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Sum>training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Shape*#
_output_shapes
:џџџџџџџџџ*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
Tshape0
К
<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/NegNegloss/dense_3_loss/mul_1*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ

Btraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDiv_1RealDiv<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Negloss/dense_3_loss/Mean_1*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ

Btraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDiv_2RealDivBtraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDiv_1loss/dense_3_loss/Mean_1*.
_class$
" loc:@loss/dense_3_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ*
T0
Є
<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/mulMul=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/truedivBtraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDiv_2*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ
Х
>training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Sum_1Sum<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/mulPtraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1
Ў
Btraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Reshape_1Reshape>training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Sum_1@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Shape_1*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
Tshape0*
_output_shapes
: 
М
:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/ShapeShapeloss/dense_3_loss/Mean*
_output_shapes
:*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*
out_type0
О
<training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Shape_1Shapedense_3_sample_weights*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*
out_type0*
_output_shapes
:
Ц
Jtraining/Adam/gradients/loss/dense_3_loss/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Shape<training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Shape_1*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ѓ
8training/Adam/gradients/loss/dense_3_loss/mul_1_grad/MulMul@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Reshapedense_3_sample_weights*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*#
_output_shapes
:џџџџџџџџџ
Б
8training/Adam/gradients/loss/dense_3_loss/mul_1_grad/SumSum8training/Adam/gradients/loss/dense_3_loss/mul_1_grad/MulJtraining/Adam/gradients/loss/dense_3_loss/mul_1_grad/BroadcastGradientArgs*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
Ѕ
<training/Adam/gradients/loss/dense_3_loss/mul_1_grad/ReshapeReshape8training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Sum:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Shape*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*
Tshape0*#
_output_shapes
:џџџџџџџџџ
ѕ
:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Mul_1Mulloss/dense_3_loss/Mean@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Reshape**
_class 
loc:@loss/dense_3_loss/mul_1*#
_output_shapes
:џџџџџџџџџ*
T0
З
:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Sum_1Sum:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Mul_1Ltraining/Adam/gradients/loss/dense_3_loss/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*
_output_shapes
:
Ћ
>training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Reshape_1Reshape:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Sum_1<training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Shape_1*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*
Tshape0*#
_output_shapes
:џџџџџџџџџ
Й
9training/Adam/gradients/loss/dense_3_loss/Mean_grad/ShapeShapeloss/dense_3_loss/Neg*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
out_type0*
_output_shapes
:
Ѕ
8training/Adam/gradients/loss/dense_3_loss/Mean_grad/SizeConst*)
_class
loc:@loss/dense_3_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
№
7training/Adam/gradients/loss/dense_3_loss/Mean_grad/addAdd(loss/dense_3_loss/Mean/reduction_indices8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Size*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 

7training/Adam/gradients/loss/dense_3_loss/Mean_grad/modFloorMod7training/Adam/gradients/loss/dense_3_loss/Mean_grad/add8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Size*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
А
;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_1Const*)
_class
loc:@loss/dense_3_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
Ќ
?training/Adam/gradients/loss/dense_3_loss/Mean_grad/range/startConst*)
_class
loc:@loss/dense_3_loss/Mean*
value	B : *
dtype0*
_output_shapes
: 
Ќ
?training/Adam/gradients/loss/dense_3_loss/Mean_grad/range/deltaConst*)
_class
loc:@loss/dense_3_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
б
9training/Adam/gradients/loss/dense_3_loss/Mean_grad/rangeRange?training/Adam/gradients/loss/dense_3_loss/Mean_grad/range/start8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Size?training/Adam/gradients/loss/dense_3_loss/Mean_grad/range/delta*
_output_shapes
:*

Tidx0*)
_class
loc:@loss/dense_3_loss/Mean
Ћ
>training/Adam/gradients/loss/dense_3_loss/Mean_grad/Fill/valueConst*)
_class
loc:@loss/dense_3_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 

8training/Adam/gradients/loss/dense_3_loss/Mean_grad/FillFill;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_1>training/Adam/gradients/loss/dense_3_loss/Mean_grad/Fill/value*
T0*)
_class
loc:@loss/dense_3_loss/Mean*

index_type0*
_output_shapes
: 

Atraining/Adam/gradients/loss/dense_3_loss/Mean_grad/DynamicStitchDynamicStitch9training/Adam/gradients/loss/dense_3_loss/Mean_grad/range7training/Adam/gradients/loss/dense_3_loss/Mean_grad/mod9training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Fill*
_output_shapes
:*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
N
Њ
=training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum/yConst*)
_class
loc:@loss/dense_3_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 

;training/Adam/gradients/loss/dense_3_loss/Mean_grad/MaximumMaximumAtraining/Adam/gradients/loss/dense_3_loss/Mean_grad/DynamicStitch=training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum/y*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:

<training/Adam/gradients/loss/dense_3_loss/Mean_grad/floordivFloorDiv9training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
Ў
;training/Adam/gradients/loss/dense_3_loss/Mean_grad/ReshapeReshape<training/Adam/gradients/loss/dense_3_loss/mul_1_grad/ReshapeAtraining/Adam/gradients/loss/dense_3_loss/Mean_grad/DynamicStitch*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
Tshape0*#
_output_shapes
:џџџџџџџџџ
І
8training/Adam/gradients/loss/dense_3_loss/Mean_grad/TileTile;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Reshape<training/Adam/gradients/loss/dense_3_loss/Mean_grad/floordiv*
T0*)
_class
loc:@loss/dense_3_loss/Mean*#
_output_shapes
:џџџџџџџџџ*

Tmultiples0
Л
;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_2Shapeloss/dense_3_loss/Neg*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
out_type0*
_output_shapes
:
М
;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_3Shapeloss/dense_3_loss/Mean*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
out_type0*
_output_shapes
:
Ў
9training/Adam/gradients/loss/dense_3_loss/Mean_grad/ConstConst*)
_class
loc:@loss/dense_3_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
Ё
8training/Adam/gradients/loss/dense_3_loss/Mean_grad/ProdProd;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_29training/Adam/gradients/loss/dense_3_loss/Mean_grad/Const*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: *
	keep_dims( *

Tidx0
А
;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Const_1Const*)
_class
loc:@loss/dense_3_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
Ѕ
:training/Adam/gradients/loss/dense_3_loss/Mean_grad/Prod_1Prod;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_3;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*)
_class
loc:@loss/dense_3_loss/Mean
Ќ
?training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum_1/yConst*)
_class
loc:@loss/dense_3_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 

=training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum_1Maximum:training/Adam/gradients/loss/dense_3_loss/Mean_grad/Prod_1?training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum_1/y*
_output_shapes
: *
T0*)
_class
loc:@loss/dense_3_loss/Mean

>training/Adam/gradients/loss/dense_3_loss/Mean_grad/floordiv_1FloorDiv8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Prod=training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum_1*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
ы
8training/Adam/gradients/loss/dense_3_loss/Mean_grad/CastCast>training/Adam/gradients/loss/dense_3_loss/Mean_grad/floordiv_1*

SrcT0*)
_class
loc:@loss/dense_3_loss/Mean*
Truncate( *
_output_shapes
: *

DstT0

;training/Adam/gradients/loss/dense_3_loss/Mean_grad/truedivRealDiv8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Tile8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Cast*
T0*)
_class
loc:@loss/dense_3_loss/Mean*#
_output_shapes
:џџџџџџџџџ
в
6training/Adam/gradients/loss/dense_3_loss/Neg_grad/NegNeg;training/Adam/gradients/loss/dense_3_loss/Mean_grad/truediv*#
_output_shapes
:џџџџџџџџџ*
T0*(
_class
loc:@loss/dense_3_loss/Neg
Л
:training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/ShapeShapeloss/dense_3_loss/mul**
_class 
loc:@loss/dense_3_loss/Sum_1*
out_type0*
_output_shapes
:*
T0
Ї
9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/SizeConst**
_class 
loc:@loss/dense_3_loss/Sum_1*
value	B :*
dtype0*
_output_shapes
: 
ђ
8training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/addAdd)loss/dense_3_loss/Sum_1/reduction_indices9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Size*
T0**
_class 
loc:@loss/dense_3_loss/Sum_1*
_output_shapes
: 

8training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/modFloorMod8training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/add9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Size*
T0**
_class 
loc:@loss/dense_3_loss/Sum_1*
_output_shapes
: 
Ћ
<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Shape_1Const**
_class 
loc:@loss/dense_3_loss/Sum_1*
valueB *
dtype0*
_output_shapes
: 
Ў
@training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/range/startConst**
_class 
loc:@loss/dense_3_loss/Sum_1*
value	B : *
dtype0*
_output_shapes
: 
Ў
@training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/range/deltaConst**
_class 
loc:@loss/dense_3_loss/Sum_1*
value	B :*
dtype0*
_output_shapes
: 
ж
:training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/rangeRange@training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/range/start9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Size@training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/range/delta**
_class 
loc:@loss/dense_3_loss/Sum_1*
_output_shapes
:*

Tidx0
­
?training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Fill/valueConst**
_class 
loc:@loss/dense_3_loss/Sum_1*
value	B :*
dtype0*
_output_shapes
: 

9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/FillFill<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Shape_1?training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Fill/value*
T0**
_class 
loc:@loss/dense_3_loss/Sum_1*

index_type0*
_output_shapes
: 

Btraining/Adam/gradients/loss/dense_3_loss/Sum_1_grad/DynamicStitchDynamicStitch:training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/range8training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/mod:training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Shape9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Fill**
_class 
loc:@loss/dense_3_loss/Sum_1*
N*
_output_shapes
:*
T0
Ќ
>training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Maximum/yConst**
_class 
loc:@loss/dense_3_loss/Sum_1*
value	B :*
dtype0*
_output_shapes
: 

<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/MaximumMaximumBtraining/Adam/gradients/loss/dense_3_loss/Sum_1_grad/DynamicStitch>training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Maximum/y*
T0**
_class 
loc:@loss/dense_3_loss/Sum_1*
_output_shapes
:

=training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/floordivFloorDiv:training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Shape<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Maximum*
T0**
_class 
loc:@loss/dense_3_loss/Sum_1*
_output_shapes
:
И
<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/ReshapeReshape6training/Adam/gradients/loss/dense_3_loss/Neg_grad/NegBtraining/Adam/gradients/loss/dense_3_loss/Sum_1_grad/DynamicStitch*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0**
_class 
loc:@loss/dense_3_loss/Sum_1*
Tshape0
Ў
9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/TileTile<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Reshape=training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/floordiv*

Tmultiples0*
T0**
_class 
loc:@loss/dense_3_loss/Sum_1*'
_output_shapes
:џџџџџџџџџ
А
8training/Adam/gradients/loss/dense_3_loss/mul_grad/ShapeShapedense_3_target*
T0*(
_class
loc:@loss/dense_3_loss/mul*
out_type0*
_output_shapes
:
Й
:training/Adam/gradients/loss/dense_3_loss/mul_grad/Shape_1Shapeloss/dense_3_loss/Log*
T0*(
_class
loc:@loss/dense_3_loss/mul*
out_type0*
_output_shapes
:
О
Htraining/Adam/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8training/Adam/gradients/loss/dense_3_loss/mul_grad/Shape:training/Adam/gradients/loss/dense_3_loss/mul_grad/Shape_1*
T0*(
_class
loc:@loss/dense_3_loss/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ы
6training/Adam/gradients/loss/dense_3_loss/mul_grad/MulMul9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Tileloss/dense_3_loss/Log*
T0*(
_class
loc:@loss/dense_3_loss/mul*'
_output_shapes
:џџџџџџџџџ
Љ
6training/Adam/gradients/loss/dense_3_loss/mul_grad/SumSum6training/Adam/gradients/loss/dense_3_loss/mul_grad/MulHtraining/Adam/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgs*(
_class
loc:@loss/dense_3_loss/mul*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Њ
:training/Adam/gradients/loss/dense_3_loss/mul_grad/ReshapeReshape6training/Adam/gradients/loss/dense_3_loss/mul_grad/Sum8training/Adam/gradients/loss/dense_3_loss/mul_grad/Shape*(
_class
loc:@loss/dense_3_loss/mul*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
ц
8training/Adam/gradients/loss/dense_3_loss/mul_grad/Mul_1Muldense_3_target9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Tile*
T0*(
_class
loc:@loss/dense_3_loss/mul*'
_output_shapes
:џџџџџџџџџ
Џ
8training/Adam/gradients/loss/dense_3_loss/mul_grad/Sum_1Sum8training/Adam/gradients/loss/dense_3_loss/mul_grad/Mul_1Jtraining/Adam/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgs:1*
T0*(
_class
loc:@loss/dense_3_loss/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
Ї
<training/Adam/gradients/loss/dense_3_loss/mul_grad/Reshape_1Reshape8training/Adam/gradients/loss/dense_3_loss/mul_grad/Sum_1:training/Adam/gradients/loss/dense_3_loss/mul_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*(
_class
loc:@loss/dense_3_loss/mul*
Tshape0

=training/Adam/gradients/loss/dense_3_loss/Log_grad/Reciprocal
Reciprocalloss/dense_3_loss/clip_by_value=^training/Adam/gradients/loss/dense_3_loss/mul_grad/Reshape_1*
T0*(
_class
loc:@loss/dense_3_loss/Log*'
_output_shapes
:џџџџџџџџџ

6training/Adam/gradients/loss/dense_3_loss/Log_grad/mulMul<training/Adam/gradients/loss/dense_3_loss/mul_grad/Reshape_1=training/Adam/gradients/loss/dense_3_loss/Log_grad/Reciprocal*
T0*(
_class
loc:@loss/dense_3_loss/Log*'
_output_shapes
:џџџџџџџџџ
н
Btraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/ShapeShape'loss/dense_3_loss/clip_by_value/Minimum*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
out_type0*
_output_shapes
:
Л
Dtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape_1Const*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
valueB *
dtype0*
_output_shapes
: 
ю
Dtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape_2Shape6training/Adam/gradients/loss/dense_3_loss/Log_grad/mul*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
out_type0*
_output_shapes
:
С
Htraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
valueB
 *    
в
Btraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/zerosFillDtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape_2Htraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/zeros/Const*'
_output_shapes
:џџџџџџџџџ*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*

index_type0

Itraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/GreaterEqualGreaterEqual'loss/dense_3_loss/clip_by_value/Minimumloss/dense_3_loss/Const*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*'
_output_shapes
:џџџџџџџџџ
ц
Rtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsBtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/ShapeDtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape_1*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
њ
Ctraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/SelectSelectItraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/GreaterEqual6training/Adam/gradients/loss/dense_3_loss/Log_grad/mulBtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/zeros*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*'
_output_shapes
:џџџџџџџџџ
ќ
Etraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Select_1SelectItraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/GreaterEqualBtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/zeros6training/Adam/gradients/loss/dense_3_loss/Log_grad/mul*'
_output_shapes
:џџџџџџџџџ*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value
д
@training/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/SumSumCtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/SelectRtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/BroadcastGradientArgs*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
_output_shapes
:*
	keep_dims( *

Tidx0
Щ
Dtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/ReshapeReshape@training/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/SumBtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
Tshape0*'
_output_shapes
:џџџџџџџџџ
к
Btraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Sum_1SumEtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Select_1Ttraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value
О
Ftraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Reshape_1ReshapeBtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Sum_1Dtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape_1*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
Tshape0*
_output_shapes
: 
п
Jtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/ShapeShapeloss/dense_3_loss/truediv*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
out_type0*
_output_shapes
:
Ы
Ltraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape_1Const*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
valueB *
dtype0*
_output_shapes
: 

Ltraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape_2ShapeDtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Reshape*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
out_type0*
_output_shapes
:
б
Ptraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/zeros/ConstConst*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
valueB
 *    *
dtype0*
_output_shapes
: 
ђ
Jtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/zerosFillLtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape_2Ptraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/zeros/Const*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*

index_type0*'
_output_shapes
:џџџџџџџџџ
ћ
Ntraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/LessEqual	LessEqualloss/dense_3_loss/truedivloss/dense_3_loss/sub*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum

Ztraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/ShapeLtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape_1*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ѕ
Ktraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/SelectSelectNtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/LessEqualDtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/ReshapeJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/zeros*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum
Ї
Mtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Select_1SelectNtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/LessEqualJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/zerosDtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Reshape*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ
є
Htraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/SumSumKtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/SelectZtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum
щ
Ltraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/ReshapeReshapeHtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/SumJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
Tshape0
њ
Jtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Sum_1SumMtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Select_1\training/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
о
Ntraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Reshape_1ReshapeJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Sum_1Ltraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape_1*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
Tshape0*
_output_shapes
: *
T0
Й
<training/Adam/gradients/loss/dense_3_loss/truediv_grad/ShapeShapedense_3/Softmax*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*
out_type0*
_output_shapes
:
С
>training/Adam/gradients/loss/dense_3_loss/truediv_grad/Shape_1Shapeloss/dense_3_loss/Sum*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*
out_type0*
_output_shapes
:
Ю
Ltraining/Adam/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs<training/Adam/gradients/loss/dense_3_loss/truediv_grad/Shape>training/Adam/gradients/loss/dense_3_loss/truediv_grad/Shape_1*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

>training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDivRealDivLtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Reshapeloss/dense_3_loss/Sum*'
_output_shapes
:џџџџџџџџџ*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv
Н
:training/Adam/gradients/loss/dense_3_loss/truediv_grad/SumSum>training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDivLtraining/Adam/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv
Б
>training/Adam/gradients/loss/dense_3_loss/truediv_grad/ReshapeReshape:training/Adam/gradients/loss/dense_3_loss/truediv_grad/Sum<training/Adam/gradients/loss/dense_3_loss/truediv_grad/Shape*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*
Tshape0*'
_output_shapes
:џџџџџџџџџ
В
:training/Adam/gradients/loss/dense_3_loss/truediv_grad/NegNegdense_3/Softmax*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*'
_output_shapes
:џџџџџџџџџ
ў
@training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDiv_1RealDiv:training/Adam/gradients/loss/dense_3_loss/truediv_grad/Negloss/dense_3_loss/Sum*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*'
_output_shapes
:џџџџџџџџџ

@training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDiv_2RealDiv@training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDiv_1loss/dense_3_loss/Sum*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*'
_output_shapes
:џџџџџџџџџ
Б
:training/Adam/gradients/loss/dense_3_loss/truediv_grad/mulMulLtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Reshape@training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDiv_2*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*'
_output_shapes
:џџџџџџџџџ
Н
<training/Adam/gradients/loss/dense_3_loss/truediv_grad/Sum_1Sum:training/Adam/gradients/loss/dense_3_loss/truediv_grad/mulNtraining/Adam/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgs:1*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
З
@training/Adam/gradients/loss/dense_3_loss/truediv_grad/Reshape_1Reshape<training/Adam/gradients/loss/dense_3_loss/truediv_grad/Sum_1>training/Adam/gradients/loss/dense_3_loss/truediv_grad/Shape_1*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Б
8training/Adam/gradients/loss/dense_3_loss/Sum_grad/ShapeShapedense_3/Softmax*
_output_shapes
:*
T0*(
_class
loc:@loss/dense_3_loss/Sum*
out_type0
Ѓ
7training/Adam/gradients/loss/dense_3_loss/Sum_grad/SizeConst*(
_class
loc:@loss/dense_3_loss/Sum*
value	B :*
dtype0*
_output_shapes
: 
ъ
6training/Adam/gradients/loss/dense_3_loss/Sum_grad/addAdd'loss/dense_3_loss/Sum/reduction_indices7training/Adam/gradients/loss/dense_3_loss/Sum_grad/Size*
T0*(
_class
loc:@loss/dense_3_loss/Sum*
_output_shapes
: 
ў
6training/Adam/gradients/loss/dense_3_loss/Sum_grad/modFloorMod6training/Adam/gradients/loss/dense_3_loss/Sum_grad/add7training/Adam/gradients/loss/dense_3_loss/Sum_grad/Size*
T0*(
_class
loc:@loss/dense_3_loss/Sum*
_output_shapes
: 
Ї
:training/Adam/gradients/loss/dense_3_loss/Sum_grad/Shape_1Const*
dtype0*
_output_shapes
: *(
_class
loc:@loss/dense_3_loss/Sum*
valueB 
Њ
>training/Adam/gradients/loss/dense_3_loss/Sum_grad/range/startConst*(
_class
loc:@loss/dense_3_loss/Sum*
value	B : *
dtype0*
_output_shapes
: 
Њ
>training/Adam/gradients/loss/dense_3_loss/Sum_grad/range/deltaConst*
_output_shapes
: *(
_class
loc:@loss/dense_3_loss/Sum*
value	B :*
dtype0
Ь
8training/Adam/gradients/loss/dense_3_loss/Sum_grad/rangeRange>training/Adam/gradients/loss/dense_3_loss/Sum_grad/range/start7training/Adam/gradients/loss/dense_3_loss/Sum_grad/Size>training/Adam/gradients/loss/dense_3_loss/Sum_grad/range/delta*(
_class
loc:@loss/dense_3_loss/Sum*
_output_shapes
:*

Tidx0
Љ
=training/Adam/gradients/loss/dense_3_loss/Sum_grad/Fill/valueConst*
dtype0*
_output_shapes
: *(
_class
loc:@loss/dense_3_loss/Sum*
value	B :

7training/Adam/gradients/loss/dense_3_loss/Sum_grad/FillFill:training/Adam/gradients/loss/dense_3_loss/Sum_grad/Shape_1=training/Adam/gradients/loss/dense_3_loss/Sum_grad/Fill/value*
_output_shapes
: *
T0*(
_class
loc:@loss/dense_3_loss/Sum*

index_type0

@training/Adam/gradients/loss/dense_3_loss/Sum_grad/DynamicStitchDynamicStitch8training/Adam/gradients/loss/dense_3_loss/Sum_grad/range6training/Adam/gradients/loss/dense_3_loss/Sum_grad/mod8training/Adam/gradients/loss/dense_3_loss/Sum_grad/Shape7training/Adam/gradients/loss/dense_3_loss/Sum_grad/Fill*(
_class
loc:@loss/dense_3_loss/Sum*
N*
_output_shapes
:*
T0
Ј
<training/Adam/gradients/loss/dense_3_loss/Sum_grad/Maximum/yConst*(
_class
loc:@loss/dense_3_loss/Sum*
value	B :*
dtype0*
_output_shapes
: 

:training/Adam/gradients/loss/dense_3_loss/Sum_grad/MaximumMaximum@training/Adam/gradients/loss/dense_3_loss/Sum_grad/DynamicStitch<training/Adam/gradients/loss/dense_3_loss/Sum_grad/Maximum/y*
T0*(
_class
loc:@loss/dense_3_loss/Sum*
_output_shapes
:

;training/Adam/gradients/loss/dense_3_loss/Sum_grad/floordivFloorDiv8training/Adam/gradients/loss/dense_3_loss/Sum_grad/Shape:training/Adam/gradients/loss/dense_3_loss/Sum_grad/Maximum*(
_class
loc:@loss/dense_3_loss/Sum*
_output_shapes
:*
T0
М
:training/Adam/gradients/loss/dense_3_loss/Sum_grad/ReshapeReshape@training/Adam/gradients/loss/dense_3_loss/truediv_grad/Reshape_1@training/Adam/gradients/loss/dense_3_loss/Sum_grad/DynamicStitch*
T0*(
_class
loc:@loss/dense_3_loss/Sum*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
І
7training/Adam/gradients/loss/dense_3_loss/Sum_grad/TileTile:training/Adam/gradients/loss/dense_3_loss/Sum_grad/Reshape;training/Adam/gradients/loss/dense_3_loss/Sum_grad/floordiv*

Tmultiples0*
T0*(
_class
loc:@loss/dense_3_loss/Sum*'
_output_shapes
:џџџџџџџџџ

training/Adam/gradients/AddNAddN>training/Adam/gradients/loss/dense_3_loss/truediv_grad/Reshape7training/Adam/gradients/loss/dense_3_loss/Sum_grad/Tile*,
_class"
 loc:@loss/dense_3_loss/truediv*
N*'
_output_shapes
:џџџџџџџџџ*
T0
М
0training/Adam/gradients/dense_3/Softmax_grad/mulMultraining/Adam/gradients/AddNdense_3/Softmax*
T0*"
_class
loc:@dense_3/Softmax*'
_output_shapes
:џџџџџџџџџ
Б
Btraining/Adam/gradients/dense_3/Softmax_grad/Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *"
_class
loc:@dense_3/Softmax*
valueB :
џџџџџџџџџ
 
0training/Adam/gradients/dense_3/Softmax_grad/SumSum0training/Adam/gradients/dense_3/Softmax_grad/mulBtraining/Adam/gradients/dense_3/Softmax_grad/Sum/reduction_indices*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0*
T0*"
_class
loc:@dense_3/Softmax
н
0training/Adam/gradients/dense_3/Softmax_grad/subSubtraining/Adam/gradients/AddN0training/Adam/gradients/dense_3/Softmax_grad/Sum*
T0*"
_class
loc:@dense_3/Softmax*'
_output_shapes
:џџџџџџџџџ
в
2training/Adam/gradients/dense_3/Softmax_grad/mul_1Mul0training/Adam/gradients/dense_3/Softmax_grad/subdense_3/Softmax*
T0*"
_class
loc:@dense_3/Softmax*'
_output_shapes
:џџџџџџџџџ
л
8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_3/Softmax_grad/mul_1*
T0*"
_class
loc:@dense_3/BiasAdd*
data_formatNHWC*
_output_shapes
:

2training/Adam/gradients/dense_3/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_3/Softmax_grad/mul_1dense_3/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_3/MatMul
ћ
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldropout_3/cond/Merge2training/Adam/gradients/dense_3/Softmax_grad/mul_1*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_3/MatMul
ћ
;training/Adam/gradients/dropout_3/cond/Merge_grad/cond_gradSwitch2training/Adam/gradients/dense_3/MatMul_grad/MatMuldropout_3/cond/pred_id*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*
T0*!
_class
loc:@dense_3/MatMul
И
 training/Adam/gradients/Switch_2Switchdense_2/Reludropout_3/cond/pred_id*
T0*
_class
loc:@dense_2/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
Є
 training/Adam/gradients/IdentityIdentity"training/Adam/gradients/Switch_2:1*
T0*
_class
loc:@dense_2/Relu*(
_output_shapes
:џџџџџџџџџ
Ђ
training/Adam/gradients/Shape_1Shape"training/Adam/gradients/Switch_2:1*
T0*
_class
loc:@dense_2/Relu*
out_type0*
_output_shapes
:
Ќ
#training/Adam/gradients/zeros/ConstConst!^training/Adam/gradients/Identity*
_class
loc:@dense_2/Relu*
valueB
 *    *
dtype0*
_output_shapes
: 
б
training/Adam/gradients/zerosFilltraining/Adam/gradients/Shape_1#training/Adam/gradients/zeros/Const*
T0*
_class
loc:@dense_2/Relu*

index_type0*(
_output_shapes
:џџџџџџџџџ

>training/Adam/gradients/dropout_3/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_3/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros*
T0*
_class
loc:@dense_2/Relu*
N**
_output_shapes
:џџџџџџџџџ: 
Ц
=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/ShapeShapedropout_3/cond/dropout/div*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*
out_type0*
_output_shapes
:
Ъ
?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Shape_1Shapedropout_3/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*
out_type0*
_output_shapes
:
в
Mtraining/Adam/gradients/dropout_3/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Shape?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul

;training/Adam/gradients/dropout_3/cond/dropout/mul_grad/MulMul=training/Adam/gradients/dropout_3/cond/Merge_grad/cond_grad:1dropout_3/cond/dropout/Floor*-
_class#
!loc:@dropout_3/cond/dropout/mul*(
_output_shapes
:џџџџџџџџџ*
T0
Н
;training/Adam/gradients/dropout_3/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_3/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_3/cond/dropout/mul_grad/BroadcastGradientArgs*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Shape*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*
Tshape0*(
_output_shapes
:џџџџџџџџџ

=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Mul_1Muldropout_3/cond/dropout/div=training/Adam/gradients/dropout_3/cond/Merge_grad/cond_grad:1*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*(
_output_shapes
:џџџџџџџџџ
У
=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_3/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*
_output_shapes
:
М
Atraining/Adam/gradients/dropout_3/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Shape_1*-
_class#
!loc:@dropout_3/cond/dropout/mul*
Tshape0*(
_output_shapes
:џџџџџџџџџ*
T0
О
=training/Adam/gradients/dropout_3/cond/dropout/div_grad/ShapeShapedropout_3/cond/mul*
T0*-
_class#
!loc:@dropout_3/cond/dropout/div*
out_type0*
_output_shapes
:
Б
?training/Adam/gradients/dropout_3/cond/dropout/div_grad/Shape_1Const*-
_class#
!loc:@dropout_3/cond/dropout/div*
valueB *
dtype0*
_output_shapes
: 
в
Mtraining/Adam/gradients/dropout_3/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_3/cond/dropout/div_grad/Shape?training/Adam/gradients/dropout_3/cond/dropout/div_grad/Shape_1*
T0*-
_class#
!loc:@dropout_3/cond/dropout/div*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

?training/Adam/gradients/dropout_3/cond/dropout/div_grad/RealDivRealDiv?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Reshape dropout_3/cond/dropout/keep_prob*
T0*-
_class#
!loc:@dropout_3/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ
С
;training/Adam/gradients/dropout_3/cond/dropout/div_grad/SumSum?training/Adam/gradients/dropout_3/cond/dropout/div_grad/RealDivMtraining/Adam/gradients/dropout_3/cond/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_3/cond/dropout/div
Ж
?training/Adam/gradients/dropout_3/cond/dropout/div_grad/ReshapeReshape;training/Adam/gradients/dropout_3/cond/dropout/div_grad/Sum=training/Adam/gradients/dropout_3/cond/dropout/div_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_3/cond/dropout/div*
Tshape0
И
;training/Adam/gradients/dropout_3/cond/dropout/div_grad/NegNegdropout_3/cond/mul*(
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_3/cond/dropout/div

Atraining/Adam/gradients/dropout_3/cond/dropout/div_grad/RealDiv_1RealDiv;training/Adam/gradients/dropout_3/cond/dropout/div_grad/Neg dropout_3/cond/dropout/keep_prob*(
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_3/cond/dropout/div

Atraining/Adam/gradients/dropout_3/cond/dropout/div_grad/RealDiv_2RealDivAtraining/Adam/gradients/dropout_3/cond/dropout/div_grad/RealDiv_1 dropout_3/cond/dropout/keep_prob*(
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_3/cond/dropout/div
Ј
;training/Adam/gradients/dropout_3/cond/dropout/div_grad/mulMul?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/ReshapeAtraining/Adam/gradients/dropout_3/cond/dropout/div_grad/RealDiv_2*(
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_3/cond/dropout/div
С
=training/Adam/gradients/dropout_3/cond/dropout/div_grad/Sum_1Sum;training/Adam/gradients/dropout_3/cond/dropout/div_grad/mulOtraining/Adam/gradients/dropout_3/cond/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_3/cond/dropout/div*
_output_shapes
:
Њ
Atraining/Adam/gradients/dropout_3/cond/dropout/div_grad/Reshape_1Reshape=training/Adam/gradients/dropout_3/cond/dropout/div_grad/Sum_1?training/Adam/gradients/dropout_3/cond/dropout/div_grad/Shape_1*
_output_shapes
: *
T0*-
_class#
!loc:@dropout_3/cond/dropout/div*
Tshape0
З
5training/Adam/gradients/dropout_3/cond/mul_grad/ShapeShapedropout_3/cond/mul/Switch:1*
T0*%
_class
loc:@dropout_3/cond/mul*
out_type0*
_output_shapes
:
Ё
7training/Adam/gradients/dropout_3/cond/mul_grad/Shape_1Const*%
_class
loc:@dropout_3/cond/mul*
valueB *
dtype0*
_output_shapes
: 
В
Etraining/Adam/gradients/dropout_3/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/dropout_3/cond/mul_grad/Shape7training/Adam/gradients/dropout_3/cond/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*%
_class
loc:@dropout_3/cond/mul
ы
3training/Adam/gradients/dropout_3/cond/mul_grad/MulMul?training/Adam/gradients/dropout_3/cond/dropout/div_grad/Reshapedropout_3/cond/mul/y*
T0*%
_class
loc:@dropout_3/cond/mul*(
_output_shapes
:џџџџџџџџџ

3training/Adam/gradients/dropout_3/cond/mul_grad/SumSum3training/Adam/gradients/dropout_3/cond/mul_grad/MulEtraining/Adam/gradients/dropout_3/cond/mul_grad/BroadcastGradientArgs*%
_class
loc:@dropout_3/cond/mul*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

7training/Adam/gradients/dropout_3/cond/mul_grad/ReshapeReshape3training/Adam/gradients/dropout_3/cond/mul_grad/Sum5training/Adam/gradients/dropout_3/cond/mul_grad/Shape*%
_class
loc:@dropout_3/cond/mul*
Tshape0*(
_output_shapes
:џџџџџџџџџ*
T0
є
5training/Adam/gradients/dropout_3/cond/mul_grad/Mul_1Muldropout_3/cond/mul/Switch:1?training/Adam/gradients/dropout_3/cond/dropout/div_grad/Reshape*%
_class
loc:@dropout_3/cond/mul*(
_output_shapes
:џџџџџџџџџ*
T0
Ѓ
5training/Adam/gradients/dropout_3/cond/mul_grad/Sum_1Sum5training/Adam/gradients/dropout_3/cond/mul_grad/Mul_1Gtraining/Adam/gradients/dropout_3/cond/mul_grad/BroadcastGradientArgs:1*%
_class
loc:@dropout_3/cond/mul*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

9training/Adam/gradients/dropout_3/cond/mul_grad/Reshape_1Reshape5training/Adam/gradients/dropout_3/cond/mul_grad/Sum_17training/Adam/gradients/dropout_3/cond/mul_grad/Shape_1*
_output_shapes
: *
T0*%
_class
loc:@dropout_3/cond/mul*
Tshape0
И
 training/Adam/gradients/Switch_3Switchdense_2/Reludropout_3/cond/pred_id*
T0*
_class
loc:@dense_2/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
Є
"training/Adam/gradients/Identity_1Identity training/Adam/gradients/Switch_3*
T0*
_class
loc:@dense_2/Relu*(
_output_shapes
:џџџџџџџџџ
 
training/Adam/gradients/Shape_2Shape training/Adam/gradients/Switch_3*
_class
loc:@dense_2/Relu*
out_type0*
_output_shapes
:*
T0
А
%training/Adam/gradients/zeros_1/ConstConst#^training/Adam/gradients/Identity_1*
_output_shapes
: *
_class
loc:@dense_2/Relu*
valueB
 *    *
dtype0
е
training/Adam/gradients/zeros_1Filltraining/Adam/gradients/Shape_2%training/Adam/gradients/zeros_1/Const*
T0*
_class
loc:@dense_2/Relu*

index_type0*(
_output_shapes
:џџџџџџџџџ

@training/Adam/gradients/dropout_3/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_17training/Adam/gradients/dropout_3/cond/mul_grad/Reshape*
T0*
_class
loc:@dense_2/Relu*
N**
_output_shapes
:џџџџџџџџџ: 

training/Adam/gradients/AddN_1AddN>training/Adam/gradients/dropout_3/cond/Switch_1_grad/cond_grad@training/Adam/gradients/dropout_3/cond/mul/Switch_grad/cond_grad*
T0*
_class
loc:@dense_2/Relu*
N*(
_output_shapes
:џџџџџџџџџ
Р
2training/Adam/gradients/dense_2/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_1dense_2/Relu*
T0*
_class
loc:@dense_2/Relu*(
_output_shapes
:џџџџџџџџџ
м
8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0*"
_class
loc:@dense_2/BiasAdd

2training/Adam/gradients/dense_2/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_2/Relu_grad/ReluGraddense_2/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_2/MatMul
ќ
4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1MatMuldropout_2/cond/Merge2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
T0*!
_class
loc:@dense_2/MatMul* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
ћ
;training/Adam/gradients/dropout_2/cond/Merge_grad/cond_gradSwitch2training/Adam/gradients/dense_2/MatMul_grad/MatMuldropout_2/cond/pred_id*
T0*!
_class
loc:@dense_2/MatMul*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
И
 training/Adam/gradients/Switch_4Switchdense_1/Reludropout_2/cond/pred_id*
T0*
_class
loc:@dense_1/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
І
"training/Adam/gradients/Identity_2Identity"training/Adam/gradients/Switch_4:1*
T0*
_class
loc:@dense_1/Relu*(
_output_shapes
:џџџџџџџџџ
Ђ
training/Adam/gradients/Shape_3Shape"training/Adam/gradients/Switch_4:1*
T0*
_class
loc:@dense_1/Relu*
out_type0*
_output_shapes
:
А
%training/Adam/gradients/zeros_2/ConstConst#^training/Adam/gradients/Identity_2*
_class
loc:@dense_1/Relu*
valueB
 *    *
dtype0*
_output_shapes
: 
е
training/Adam/gradients/zeros_2Filltraining/Adam/gradients/Shape_3%training/Adam/gradients/zeros_2/Const*
T0*
_class
loc:@dense_1/Relu*

index_type0*(
_output_shapes
:џџџџџџџџџ

>training/Adam/gradients/dropout_2/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_2/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros_2*
_class
loc:@dense_1/Relu*
N**
_output_shapes
:џџџџџџџџџ: *
T0
Ц
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ShapeShapedropout_2/cond/dropout/div*
_output_shapes
:*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
out_type0
Ъ
?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1Shapedropout_2/cond/dropout/Floor*-
_class#
!loc:@dropout_2/cond/dropout/mul*
out_type0*
_output_shapes
:*
T0
в
Mtraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/MulMul=training/Adam/gradients/dropout_2/cond/Merge_grad/cond_grad:1dropout_2/cond/dropout/Floor*(
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul
Н
;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
_output_shapes
:
Ж
?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
Tshape0*(
_output_shapes
:џџџџџџџџџ

=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Muldropout_2/cond/dropout/div=training/Adam/gradients/dropout_2/cond/Merge_grad/cond_grad:1*(
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul
У
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
_output_shapes
:
М
Atraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*(
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
Tshape0
О
=training/Adam/gradients/dropout_2/cond/dropout/div_grad/ShapeShapedropout_2/cond/mul*
_output_shapes
:*
T0*-
_class#
!loc:@dropout_2/cond/dropout/div*
out_type0
Б
?training/Adam/gradients/dropout_2/cond/dropout/div_grad/Shape_1Const*-
_class#
!loc:@dropout_2/cond/dropout/div*
valueB *
dtype0*
_output_shapes
: 
в
Mtraining/Adam/gradients/dropout_2/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_2/cond/dropout/div_grad/Shape?training/Adam/gradients/dropout_2/cond/dropout/div_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_2/cond/dropout/div

?training/Adam/gradients/dropout_2/cond/dropout/div_grad/RealDivRealDiv?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Reshape dropout_2/cond/dropout/keep_prob*-
_class#
!loc:@dropout_2/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ*
T0
С
;training/Adam/gradients/dropout_2/cond/dropout/div_grad/SumSum?training/Adam/gradients/dropout_2/cond/dropout/div_grad/RealDivMtraining/Adam/gradients/dropout_2/cond/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_2/cond/dropout/div*
_output_shapes
:
Ж
?training/Adam/gradients/dropout_2/cond/dropout/div_grad/ReshapeReshape;training/Adam/gradients/dropout_2/cond/dropout/div_grad/Sum=training/Adam/gradients/dropout_2/cond/dropout/div_grad/Shape*-
_class#
!loc:@dropout_2/cond/dropout/div*
Tshape0*(
_output_shapes
:џџџџџџџџџ*
T0
И
;training/Adam/gradients/dropout_2/cond/dropout/div_grad/NegNegdropout_2/cond/mul*
T0*-
_class#
!loc:@dropout_2/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ

Atraining/Adam/gradients/dropout_2/cond/dropout/div_grad/RealDiv_1RealDiv;training/Adam/gradients/dropout_2/cond/dropout/div_grad/Neg dropout_2/cond/dropout/keep_prob*
T0*-
_class#
!loc:@dropout_2/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ

Atraining/Adam/gradients/dropout_2/cond/dropout/div_grad/RealDiv_2RealDivAtraining/Adam/gradients/dropout_2/cond/dropout/div_grad/RealDiv_1 dropout_2/cond/dropout/keep_prob*
T0*-
_class#
!loc:@dropout_2/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ
Ј
;training/Adam/gradients/dropout_2/cond/dropout/div_grad/mulMul?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ReshapeAtraining/Adam/gradients/dropout_2/cond/dropout/div_grad/RealDiv_2*
T0*-
_class#
!loc:@dropout_2/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ
С
=training/Adam/gradients/dropout_2/cond/dropout/div_grad/Sum_1Sum;training/Adam/gradients/dropout_2/cond/dropout/div_grad/mulOtraining/Adam/gradients/dropout_2/cond/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_2/cond/dropout/div*
_output_shapes
:
Њ
Atraining/Adam/gradients/dropout_2/cond/dropout/div_grad/Reshape_1Reshape=training/Adam/gradients/dropout_2/cond/dropout/div_grad/Sum_1?training/Adam/gradients/dropout_2/cond/dropout/div_grad/Shape_1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/div*
Tshape0*
_output_shapes
: 
З
5training/Adam/gradients/dropout_2/cond/mul_grad/ShapeShapedropout_2/cond/mul/Switch:1*
_output_shapes
:*
T0*%
_class
loc:@dropout_2/cond/mul*
out_type0
Ё
7training/Adam/gradients/dropout_2/cond/mul_grad/Shape_1Const*
_output_shapes
: *%
_class
loc:@dropout_2/cond/mul*
valueB *
dtype0
В
Etraining/Adam/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/dropout_2/cond/mul_grad/Shape7training/Adam/gradients/dropout_2/cond/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*%
_class
loc:@dropout_2/cond/mul
ы
3training/Adam/gradients/dropout_2/cond/mul_grad/MulMul?training/Adam/gradients/dropout_2/cond/dropout/div_grad/Reshapedropout_2/cond/mul/y*(
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@dropout_2/cond/mul

3training/Adam/gradients/dropout_2/cond/mul_grad/SumSum3training/Adam/gradients/dropout_2/cond/mul_grad/MulEtraining/Adam/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgs*
T0*%
_class
loc:@dropout_2/cond/mul*
_output_shapes
:*
	keep_dims( *

Tidx0

7training/Adam/gradients/dropout_2/cond/mul_grad/ReshapeReshape3training/Adam/gradients/dropout_2/cond/mul_grad/Sum5training/Adam/gradients/dropout_2/cond/mul_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@dropout_2/cond/mul*
Tshape0
є
5training/Adam/gradients/dropout_2/cond/mul_grad/Mul_1Muldropout_2/cond/mul/Switch:1?training/Adam/gradients/dropout_2/cond/dropout/div_grad/Reshape*
T0*%
_class
loc:@dropout_2/cond/mul*(
_output_shapes
:џџџџџџџџџ
Ѓ
5training/Adam/gradients/dropout_2/cond/mul_grad/Sum_1Sum5training/Adam/gradients/dropout_2/cond/mul_grad/Mul_1Gtraining/Adam/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@dropout_2/cond/mul*
_output_shapes
:

9training/Adam/gradients/dropout_2/cond/mul_grad/Reshape_1Reshape5training/Adam/gradients/dropout_2/cond/mul_grad/Sum_17training/Adam/gradients/dropout_2/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_2/cond/mul*
Tshape0*
_output_shapes
: 
И
 training/Adam/gradients/Switch_5Switchdense_1/Reludropout_2/cond/pred_id*
T0*
_class
loc:@dense_1/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
Є
"training/Adam/gradients/Identity_3Identity training/Adam/gradients/Switch_5*
T0*
_class
loc:@dense_1/Relu*(
_output_shapes
:џџџџџџџџџ
 
training/Adam/gradients/Shape_4Shape training/Adam/gradients/Switch_5*
T0*
_class
loc:@dense_1/Relu*
out_type0*
_output_shapes
:
А
%training/Adam/gradients/zeros_3/ConstConst#^training/Adam/gradients/Identity_3*
_output_shapes
: *
_class
loc:@dense_1/Relu*
valueB
 *    *
dtype0
е
training/Adam/gradients/zeros_3Filltraining/Adam/gradients/Shape_4%training/Adam/gradients/zeros_3/Const*
_class
loc:@dense_1/Relu*

index_type0*(
_output_shapes
:џџџџџџџџџ*
T0

@training/Adam/gradients/dropout_2/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_37training/Adam/gradients/dropout_2/cond/mul_grad/Reshape*
T0*
_class
loc:@dense_1/Relu*
N**
_output_shapes
:џџџџџџџџџ: 

training/Adam/gradients/AddN_2AddN>training/Adam/gradients/dropout_2/cond/Switch_1_grad/cond_grad@training/Adam/gradients/dropout_2/cond/mul/Switch_grad/cond_grad*
T0*
_class
loc:@dense_1/Relu*
N*(
_output_shapes
:џџџџџџџџџ
Р
2training/Adam/gradients/dense_1/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_2dense_1/Relu*
_class
loc:@dense_1/Relu*(
_output_shapes
:џџџџџџџџџ*
T0
м
8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_1/Relu_grad/ReluGrad*
_output_shapes	
:*
T0*"
_class
loc:@dense_1/BiasAdd*
data_formatNHWC

2training/Adam/gradients/dense_1/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_1/Relu_grad/ReluGraddense_1/kernel/read*
transpose_b(*
T0*!
_class
loc:@dense_1/MatMul*(
_output_shapes
:џџџџџџџџџЪ*
transpose_a( 
љ
4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1MatMulflatten_1/Reshape2training/Adam/gradients/dense_1/Relu_grad/ReluGrad*
transpose_b( *
T0*!
_class
loc:@dense_1/MatMul* 
_output_shapes
:
Ъ*
transpose_a(
Ў
4training/Adam/gradients/flatten_1/Reshape_grad/ShapeShapedropout_1/cond/Merge*
T0*$
_class
loc:@flatten_1/Reshape*
out_type0*
_output_shapes
:

6training/Adam/gradients/flatten_1/Reshape_grad/ReshapeReshape2training/Adam/gradients/dense_1/MatMul_grad/MatMul4training/Adam/gradients/flatten_1/Reshape_grad/Shape*
T0*$
_class
loc:@flatten_1/Reshape*
Tshape0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

;training/Adam/gradients/dropout_1/cond/Merge_grad/cond_gradSwitch6training/Adam/gradients/flatten_1/Reshape_grad/Reshapedropout_1/cond/pred_id*
T0*$
_class
loc:@flatten_1/Reshape*T
_output_shapesB
@:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ
м
 training/Adam/gradients/Switch_6Switchlstm_1/transpose_1dropout_1/cond/pred_id*T
_output_shapesB
@:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ*
T0*%
_class
loc:@lstm_1/transpose_1
И
"training/Adam/gradients/Identity_4Identity"training/Adam/gradients/Switch_6:1*
T0*%
_class
loc:@lstm_1/transpose_1*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
Ј
training/Adam/gradients/Shape_5Shape"training/Adam/gradients/Switch_6:1*
_output_shapes
:*
T0*%
_class
loc:@lstm_1/transpose_1*
out_type0
Ж
%training/Adam/gradients/zeros_4/ConstConst#^training/Adam/gradients/Identity_4*%
_class
loc:@lstm_1/transpose_1*
valueB
 *    *
dtype0*
_output_shapes
: 
ч
training/Adam/gradients/zeros_4Filltraining/Adam/gradients/Shape_5%training/Adam/gradients/zeros_4/Const*
T0*%
_class
loc:@lstm_1/transpose_1*

index_type0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

>training/Adam/gradients/dropout_1/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_1/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros_4*%
_class
loc:@lstm_1/transpose_1*
N*6
_output_shapes$
":џџџџџџџџџџџџџџџџџџ: *
T0
Ц
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ShapeShapedropout_1/cond/dropout/div*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
out_type0*
_output_shapes
:
Ъ
?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1Shapedropout_1/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
out_type0*
_output_shapes
:
в
Mtraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul

;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/MulMul=training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad:1dropout_1/cond/dropout/Floor*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul
Н
;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
Т
?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
Tshape0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Muldropout_1/cond/dropout/div=training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad:1*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul
У
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul
Ш
Atraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
Tshape0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
О
=training/Adam/gradients/dropout_1/cond/dropout/div_grad/ShapeShapedropout_1/cond/mul*-
_class#
!loc:@dropout_1/cond/dropout/div*
out_type0*
_output_shapes
:*
T0
Б
?training/Adam/gradients/dropout_1/cond/dropout/div_grad/Shape_1Const*-
_class#
!loc:@dropout_1/cond/dropout/div*
valueB *
dtype0*
_output_shapes
: 
в
Mtraining/Adam/gradients/dropout_1/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_1/cond/dropout/div_grad/Shape?training/Adam/gradients/dropout_1/cond/dropout/div_grad/Shape_1*-
_class#
!loc:@dropout_1/cond/dropout/div*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

?training/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDivRealDiv?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Reshape dropout_1/cond/dropout/keep_prob*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div
С
;training/Adam/gradients/dropout_1/cond/dropout/div_grad/SumSum?training/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDivMtraining/Adam/gradients/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div
Т
?training/Adam/gradients/dropout_1/cond/dropout/div_grad/ReshapeReshape;training/Adam/gradients/dropout_1/cond/dropout/div_grad/Sum=training/Adam/gradients/dropout_1/cond/dropout/div_grad/Shape*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*
Tshape0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
Ф
;training/Adam/gradients/dropout_1/cond/dropout/div_grad/NegNegdropout_1/cond/mul*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

Atraining/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDiv_1RealDiv;training/Adam/gradients/dropout_1/cond/dropout/div_grad/Neg dropout_1/cond/dropout/keep_prob*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

Atraining/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDiv_2RealDivAtraining/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDiv_1 dropout_1/cond/dropout/keep_prob*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
Д
;training/Adam/gradients/dropout_1/cond/dropout/div_grad/mulMul?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ReshapeAtraining/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDiv_2*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
С
=training/Adam/gradients/dropout_1/cond/dropout/div_grad/Sum_1Sum;training/Adam/gradients/dropout_1/cond/dropout/div_grad/mulOtraining/Adam/gradients/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div
Њ
Atraining/Adam/gradients/dropout_1/cond/dropout/div_grad/Reshape_1Reshape=training/Adam/gradients/dropout_1/cond/dropout/div_grad/Sum_1?training/Adam/gradients/dropout_1/cond/dropout/div_grad/Shape_1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*
Tshape0*
_output_shapes
: 
З
5training/Adam/gradients/dropout_1/cond/mul_grad/ShapeShapedropout_1/cond/mul/Switch:1*%
_class
loc:@dropout_1/cond/mul*
out_type0*
_output_shapes
:*
T0
Ё
7training/Adam/gradients/dropout_1/cond/mul_grad/Shape_1Const*%
_class
loc:@dropout_1/cond/mul*
valueB *
dtype0*
_output_shapes
: 
В
Etraining/Adam/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/dropout_1/cond/mul_grad/Shape7training/Adam/gradients/dropout_1/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_1/cond/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ї
3training/Adam/gradients/dropout_1/cond/mul_grad/MulMul?training/Adam/gradients/dropout_1/cond/dropout/div_grad/Reshapedropout_1/cond/mul/y*
T0*%
_class
loc:@dropout_1/cond/mul*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

3training/Adam/gradients/dropout_1/cond/mul_grad/SumSum3training/Adam/gradients/dropout_1/cond/mul_grad/MulEtraining/Adam/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@dropout_1/cond/mul*
_output_shapes
:
Ђ
7training/Adam/gradients/dropout_1/cond/mul_grad/ReshapeReshape3training/Adam/gradients/dropout_1/cond/mul_grad/Sum5training/Adam/gradients/dropout_1/cond/mul_grad/Shape*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0*%
_class
loc:@dropout_1/cond/mul*
Tshape0

5training/Adam/gradients/dropout_1/cond/mul_grad/Mul_1Muldropout_1/cond/mul/Switch:1?training/Adam/gradients/dropout_1/cond/dropout/div_grad/Reshape*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0*%
_class
loc:@dropout_1/cond/mul
Ѓ
5training/Adam/gradients/dropout_1/cond/mul_grad/Sum_1Sum5training/Adam/gradients/dropout_1/cond/mul_grad/Mul_1Gtraining/Adam/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@dropout_1/cond/mul

9training/Adam/gradients/dropout_1/cond/mul_grad/Reshape_1Reshape5training/Adam/gradients/dropout_1/cond/mul_grad/Sum_17training/Adam/gradients/dropout_1/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_1/cond/mul*
Tshape0*
_output_shapes
: 
м
 training/Adam/gradients/Switch_7Switchlstm_1/transpose_1dropout_1/cond/pred_id*T
_output_shapesB
@:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ*
T0*%
_class
loc:@lstm_1/transpose_1
Ж
"training/Adam/gradients/Identity_5Identity training/Adam/gradients/Switch_7*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0*%
_class
loc:@lstm_1/transpose_1
І
training/Adam/gradients/Shape_6Shape training/Adam/gradients/Switch_7*
_output_shapes
:*
T0*%
_class
loc:@lstm_1/transpose_1*
out_type0
Ж
%training/Adam/gradients/zeros_5/ConstConst#^training/Adam/gradients/Identity_5*%
_class
loc:@lstm_1/transpose_1*
valueB
 *    *
dtype0*
_output_shapes
: 
ч
training/Adam/gradients/zeros_5Filltraining/Adam/gradients/Shape_6%training/Adam/gradients/zeros_5/Const*
T0*%
_class
loc:@lstm_1/transpose_1*

index_type0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

@training/Adam/gradients/dropout_1/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_57training/Adam/gradients/dropout_1/cond/mul_grad/Reshape*
T0*%
_class
loc:@lstm_1/transpose_1*
N*6
_output_shapes$
":џџџџџџџџџџџџџџџџџџ: 

training/Adam/gradients/AddN_3AddN>training/Adam/gradients/dropout_1/cond/Switch_1_grad/cond_grad@training/Adam/gradients/dropout_1/cond/mul/Switch_grad/cond_grad*
T0*%
_class
loc:@lstm_1/transpose_1*
N*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
Л
Atraining/Adam/gradients/lstm_1/transpose_1_grad/InvertPermutationInvertPermutationlstm_1/transpose_1/perm*%
_class
loc:@lstm_1/transpose_1*
_output_shapes
:*
T0

9training/Adam/gradients/lstm_1/transpose_1_grad/transpose	Transposetraining/Adam/gradients/AddN_3Atraining/Adam/gradients/lstm_1/transpose_1_grad/InvertPermutation*
T0*%
_class
loc:@lstm_1/transpose_1*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
Tperm0

jtraining/Adam/gradients/lstm_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lstm_1/TensorArraylstm_1/while/Exit_2*%
_class
loc:@lstm_1/TensorArray*#
sourcetraining/Adam/gradients*
_output_shapes

:: 
М
ftraining/Adam/gradients/lstm_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentitylstm_1/while/Exit_2k^training/Adam/gradients/lstm_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*%
_class
loc:@lstm_1/TensorArray*
_output_shapes
: 
ў
ptraining/Adam/gradients/lstm_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3jtraining/Adam/gradients/lstm_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3lstm_1/TensorArrayStack/range9training/Adam/gradients/lstm_1/transpose_1_grad/transposeftraining/Adam/gradients/lstm_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*%
_class
loc:@lstm_1/TensorArray*
_output_shapes
: 
v
"training/Adam/gradients/zeros_like	ZerosLikelstm_1/while/Exit_3*
T0*'
_output_shapes
:џџџџџџџџџ
x
$training/Adam/gradients/zeros_like_1	ZerosLikelstm_1/while/Exit_4*
T0*'
_output_shapes
:џџџџџџџџџ
э
7training/Adam/gradients/lstm_1/while/Exit_2_grad/b_exitEnterptraining/Adam/gradients/lstm_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
_output_shapes
: *B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*&
_class
loc:@lstm_1/while/Exit_2*
parallel_iterations *
is_constant( 
А
7training/Adam/gradients/lstm_1/while/Exit_3_grad/b_exitEnter"training/Adam/gradients/zeros_like*
is_constant( *'
_output_shapes
:џџџџџџџџџ*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*&
_class
loc:@lstm_1/while/Exit_3*
parallel_iterations 
В
7training/Adam/gradients/lstm_1/while/Exit_4_grad/b_exitEnter$training/Adam/gradients/zeros_like_1*
T0*&
_class
loc:@lstm_1/while/Exit_4*
parallel_iterations *
is_constant( *'
_output_shapes
:џџџџџџџџџ*B

frame_name42training/Adam/gradients/lstm_1/while/while_context

;training/Adam/gradients/lstm_1/while/Switch_2_grad/b_switchMerge7training/Adam/gradients/lstm_1/while/Exit_2_grad/b_exitBtraining/Adam/gradients/lstm_1/while/Switch_2_grad_1/NextIteration*'
_class
loc:@lstm_1/while/Merge_2*
N*
_output_shapes
: : *
T0
Ї
;training/Adam/gradients/lstm_1/while/Switch_3_grad/b_switchMerge7training/Adam/gradients/lstm_1/while/Exit_3_grad/b_exitBtraining/Adam/gradients/lstm_1/while/Switch_3_grad_1/NextIteration*'
_class
loc:@lstm_1/while/Merge_3*
N*)
_output_shapes
:џџџџџџџџџ: *
T0
Ї
;training/Adam/gradients/lstm_1/while/Switch_4_grad/b_switchMerge7training/Adam/gradients/lstm_1/while/Exit_4_grad/b_exitBtraining/Adam/gradients/lstm_1/while/Switch_4_grad_1/NextIteration*
T0*'
_class
loc:@lstm_1/while/Merge_4*
N*)
_output_shapes
:џџџџџџџџџ: 
ю
8training/Adam/gradients/lstm_1/while/Merge_2_grad/SwitchSwitch;training/Adam/gradients/lstm_1/while/Switch_2_grad/b_switch!training/Adam/gradients/b_count_2*
T0*'
_class
loc:@lstm_1/while/Merge_2*
_output_shapes
: : 

8training/Adam/gradients/lstm_1/while/Merge_3_grad/SwitchSwitch;training/Adam/gradients/lstm_1/while/Switch_3_grad/b_switch!training/Adam/gradients/b_count_2*
T0*'
_class
loc:@lstm_1/while/Merge_3*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ

8training/Adam/gradients/lstm_1/while/Merge_4_grad/SwitchSwitch;training/Adam/gradients/lstm_1/while/Switch_4_grad/b_switch!training/Adam/gradients/b_count_2*
T0*'
_class
loc:@lstm_1/while/Merge_4*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
Т
6training/Adam/gradients/lstm_1/while/Enter_2_grad/ExitExit8training/Adam/gradients/lstm_1/while/Merge_2_grad/Switch*
T0*'
_class
loc:@lstm_1/while/Enter_2*
_output_shapes
: 
г
6training/Adam/gradients/lstm_1/while/Enter_3_grad/ExitExit8training/Adam/gradients/lstm_1/while/Merge_3_grad/Switch*
T0*'
_class
loc:@lstm_1/while/Enter_3*'
_output_shapes
:џџџџџџџџџ
г
6training/Adam/gradients/lstm_1/while/Enter_4_grad/ExitExit8training/Adam/gradients/lstm_1/while/Merge_4_grad/Switch*'
_output_shapes
:џџџџџџџџџ*
T0*'
_class
loc:@lstm_1/while/Enter_4
Ё
otraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3utraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter:training/Adam/gradients/lstm_1/while/Merge_2_grad/Switch:1*
_output_shapes

:: *%
_class
loc:@lstm_1/while/mul_5*#
sourcetraining/Adam/gradients
а
utraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlstm_1/TensorArray*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations *
is_constant(
э
ktraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity:training/Adam/gradients/lstm_1/while/Merge_2_grad/Switch:1p^training/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
: 

_training/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3otraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3jtraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2ktraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*%
_class
loc:@lstm_1/while/mul_5*
dtype0*'
_output_shapes
:џџџџџџџџџ
ѕ
etraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*C
_class9
7loc:@lstm_1/while/Identity_1loc:@lstm_1/while/mul_5*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
л
etraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2etraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*C
_class9
7loc:@lstm_1/while/Identity_1loc:@lstm_1/while/mul_5*

stack_name *
_output_shapes
:*
	elem_type0
ћ
etraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnteretraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
ѓ
ktraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2etraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterlstm_1/while/Identity_1^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
: *
swap_memory(
и
jtraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2ptraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
: *
	elem_type0

ptraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnteretraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations 
р
ftraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerG^training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV2G^training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV2k^training/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Q^training/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2_1Q^training/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopV2Q^training/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2_1Q^training/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopV2Q^training/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2_1Q^training/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2_1Q^training/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2_1Q^training/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPopV2a^training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopV2U^training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopV2c^training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopV2W^training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopV2[^training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopV2R^training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopV2c^training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopV2W^training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopV2[^training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopV2R^training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopV2Y^training/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopV2P^training/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopV2Q^training/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopV2?^training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPopV2Q^training/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2_1?^training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPopV2A^training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPopV2Q^training/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2_1?^training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPopV2A^training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPopV2Q^training/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopV2?^training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPopV2Q^training/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2_1?^training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV2A^training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPopV2O^training/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopV2=^training/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPopV2*%
_class
loc:@lstm_1/while/mul_5

.training/Adam/gradients/lstm_1/Tile_grad/ShapeShapelstm_1/ExpandDims*
_output_shapes
:*
T0*
_class
loc:@lstm_1/Tile*
out_type0
л
.training/Adam/gradients/lstm_1/Tile_grad/stackPacklstm_1/Tile/multiples.training/Adam/gradients/lstm_1/Tile_grad/Shape*
T0*
_class
loc:@lstm_1/Tile*

axis *
N*
_output_shapes

:
А
7training/Adam/gradients/lstm_1/Tile_grad/transpose/RankRank.training/Adam/gradients/lstm_1/Tile_grad/stack*
T0*
_class
loc:@lstm_1/Tile*
_output_shapes
: 

8training/Adam/gradients/lstm_1/Tile_grad/transpose/sub/yConst*
_class
loc:@lstm_1/Tile*
value	B :*
dtype0*
_output_shapes
: 
ё
6training/Adam/gradients/lstm_1/Tile_grad/transpose/subSub7training/Adam/gradients/lstm_1/Tile_grad/transpose/Rank8training/Adam/gradients/lstm_1/Tile_grad/transpose/sub/y*
_output_shapes
: *
T0*
_class
loc:@lstm_1/Tile
 
>training/Adam/gradients/lstm_1/Tile_grad/transpose/Range/startConst*
_class
loc:@lstm_1/Tile*
value	B : *
dtype0*
_output_shapes
: 
 
>training/Adam/gradients/lstm_1/Tile_grad/transpose/Range/deltaConst*
_class
loc:@lstm_1/Tile*
value	B :*
dtype0*
_output_shapes
: 
Ы
8training/Adam/gradients/lstm_1/Tile_grad/transpose/RangeRange>training/Adam/gradients/lstm_1/Tile_grad/transpose/Range/start7training/Adam/gradients/lstm_1/Tile_grad/transpose/Rank>training/Adam/gradients/lstm_1/Tile_grad/transpose/Range/delta*
_class
loc:@lstm_1/Tile*#
_output_shapes
:џџџџџџџџџ*

Tidx0
џ
8training/Adam/gradients/lstm_1/Tile_grad/transpose/sub_1Sub6training/Adam/gradients/lstm_1/Tile_grad/transpose/sub8training/Adam/gradients/lstm_1/Tile_grad/transpose/Range*
T0*
_class
loc:@lstm_1/Tile*#
_output_shapes
:џџџџџџџџџ
џ
2training/Adam/gradients/lstm_1/Tile_grad/transpose	Transpose.training/Adam/gradients/lstm_1/Tile_grad/stack8training/Adam/gradients/lstm_1/Tile_grad/transpose/sub_1*
Tperm0*
T0*
_class
loc:@lstm_1/Tile*
_output_shapes

:
Љ
6training/Adam/gradients/lstm_1/Tile_grad/Reshape/shapeConst*
_class
loc:@lstm_1/Tile*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
њ
0training/Adam/gradients/lstm_1/Tile_grad/ReshapeReshape2training/Adam/gradients/lstm_1/Tile_grad/transpose6training/Adam/gradients/lstm_1/Tile_grad/Reshape/shape*
T0*
_class
loc:@lstm_1/Tile*
Tshape0*
_output_shapes
:

-training/Adam/gradients/lstm_1/Tile_grad/SizeConst*
_class
loc:@lstm_1/Tile*
value	B :*
dtype0*
_output_shapes
: 

4training/Adam/gradients/lstm_1/Tile_grad/range/startConst*
_output_shapes
: *
_class
loc:@lstm_1/Tile*
value	B : *
dtype0

4training/Adam/gradients/lstm_1/Tile_grad/range/deltaConst*
_class
loc:@lstm_1/Tile*
value	B :*
dtype0*
_output_shapes
: 

.training/Adam/gradients/lstm_1/Tile_grad/rangeRange4training/Adam/gradients/lstm_1/Tile_grad/range/start-training/Adam/gradients/lstm_1/Tile_grad/Size4training/Adam/gradients/lstm_1/Tile_grad/range/delta*
_class
loc:@lstm_1/Tile*
_output_shapes
:*

Tidx0
Њ
2training/Adam/gradients/lstm_1/Tile_grad/Reshape_1Reshape6training/Adam/gradients/lstm_1/while/Enter_3_grad/Exit0training/Adam/gradients/lstm_1/Tile_grad/Reshape*
T0*
_class
loc:@lstm_1/Tile*
Tshape0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ

,training/Adam/gradients/lstm_1/Tile_grad/SumSum2training/Adam/gradients/lstm_1/Tile_grad/Reshape_1.training/Adam/gradients/lstm_1/Tile_grad/range*'
_output_shapes
:џџџџџџџџџ*
	keep_dims( *

Tidx0*
T0*
_class
loc:@lstm_1/Tile
Ѓ
0training/Adam/gradients/lstm_1/Tile_1_grad/ShapeShapelstm_1/ExpandDims*
T0* 
_class
loc:@lstm_1/Tile_1*
out_type0*
_output_shapes
:
у
0training/Adam/gradients/lstm_1/Tile_1_grad/stackPacklstm_1/Tile_1/multiples0training/Adam/gradients/lstm_1/Tile_1_grad/Shape*
T0* 
_class
loc:@lstm_1/Tile_1*

axis *
N*
_output_shapes

:
Ж
9training/Adam/gradients/lstm_1/Tile_1_grad/transpose/RankRank0training/Adam/gradients/lstm_1/Tile_1_grad/stack*
_output_shapes
: *
T0* 
_class
loc:@lstm_1/Tile_1

:training/Adam/gradients/lstm_1/Tile_1_grad/transpose/sub/yConst* 
_class
loc:@lstm_1/Tile_1*
value	B :*
dtype0*
_output_shapes
: 
љ
8training/Adam/gradients/lstm_1/Tile_1_grad/transpose/subSub9training/Adam/gradients/lstm_1/Tile_1_grad/transpose/Rank:training/Adam/gradients/lstm_1/Tile_1_grad/transpose/sub/y* 
_class
loc:@lstm_1/Tile_1*
_output_shapes
: *
T0
Є
@training/Adam/gradients/lstm_1/Tile_1_grad/transpose/Range/startConst* 
_class
loc:@lstm_1/Tile_1*
value	B : *
dtype0*
_output_shapes
: 
Є
@training/Adam/gradients/lstm_1/Tile_1_grad/transpose/Range/deltaConst* 
_class
loc:@lstm_1/Tile_1*
value	B :*
dtype0*
_output_shapes
: 
е
:training/Adam/gradients/lstm_1/Tile_1_grad/transpose/RangeRange@training/Adam/gradients/lstm_1/Tile_1_grad/transpose/Range/start9training/Adam/gradients/lstm_1/Tile_1_grad/transpose/Rank@training/Adam/gradients/lstm_1/Tile_1_grad/transpose/Range/delta*

Tidx0* 
_class
loc:@lstm_1/Tile_1*#
_output_shapes
:џџџџџџџџџ

:training/Adam/gradients/lstm_1/Tile_1_grad/transpose/sub_1Sub8training/Adam/gradients/lstm_1/Tile_1_grad/transpose/sub:training/Adam/gradients/lstm_1/Tile_1_grad/transpose/Range*#
_output_shapes
:џџџџџџџџџ*
T0* 
_class
loc:@lstm_1/Tile_1

4training/Adam/gradients/lstm_1/Tile_1_grad/transpose	Transpose0training/Adam/gradients/lstm_1/Tile_1_grad/stack:training/Adam/gradients/lstm_1/Tile_1_grad/transpose/sub_1*
_output_shapes

:*
Tperm0*
T0* 
_class
loc:@lstm_1/Tile_1
­
8training/Adam/gradients/lstm_1/Tile_1_grad/Reshape/shapeConst* 
_class
loc:@lstm_1/Tile_1*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:

2training/Adam/gradients/lstm_1/Tile_1_grad/ReshapeReshape4training/Adam/gradients/lstm_1/Tile_1_grad/transpose8training/Adam/gradients/lstm_1/Tile_1_grad/Reshape/shape*
T0* 
_class
loc:@lstm_1/Tile_1*
Tshape0*
_output_shapes
:

/training/Adam/gradients/lstm_1/Tile_1_grad/SizeConst* 
_class
loc:@lstm_1/Tile_1*
value	B :*
dtype0*
_output_shapes
: 

6training/Adam/gradients/lstm_1/Tile_1_grad/range/startConst*
dtype0*
_output_shapes
: * 
_class
loc:@lstm_1/Tile_1*
value	B : 

6training/Adam/gradients/lstm_1/Tile_1_grad/range/deltaConst* 
_class
loc:@lstm_1/Tile_1*
value	B :*
dtype0*
_output_shapes
: 
Є
0training/Adam/gradients/lstm_1/Tile_1_grad/rangeRange6training/Adam/gradients/lstm_1/Tile_1_grad/range/start/training/Adam/gradients/lstm_1/Tile_1_grad/Size6training/Adam/gradients/lstm_1/Tile_1_grad/range/delta*
_output_shapes
:*

Tidx0* 
_class
loc:@lstm_1/Tile_1
А
4training/Adam/gradients/lstm_1/Tile_1_grad/Reshape_1Reshape6training/Adam/gradients/lstm_1/while/Enter_4_grad/Exit2training/Adam/gradients/lstm_1/Tile_1_grad/Reshape*
T0* 
_class
loc:@lstm_1/Tile_1*
Tshape0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ

.training/Adam/gradients/lstm_1/Tile_1_grad/SumSum4training/Adam/gradients/lstm_1/Tile_1_grad/Reshape_10training/Adam/gradients/lstm_1/Tile_1_grad/range*
T0* 
_class
loc:@lstm_1/Tile_1*'
_output_shapes
:џџџџџџџџџ*
	keep_dims( *

Tidx0
Ї
training/Adam/gradients/AddN_4AddN:training/Adam/gradients/lstm_1/while/Merge_3_grad/Switch:1_training/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*'
_class
loc:@lstm_1/while/Merge_3*
N*'
_output_shapes
:џџџџџџџџџ*
T0
И
5training/Adam/gradients/lstm_1/while/mul_5_grad/ShapeShapelstm_1/while/clip_by_value_2*
T0*%
_class
loc:@lstm_1/while/mul_5*
out_type0*
_output_shapes
:
Б
7training/Adam/gradients/lstm_1/while/mul_5_grad/Shape_1Shapelstm_1/while/Tanh_1*
T0*%
_class
loc:@lstm_1/while/mul_5*
out_type0*
_output_shapes
:
ш
Etraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2Rtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/mul_5
Н
Ktraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/mul_5*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/mul_5*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations *
is_constant(
с
Qtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/mul_5_grad/Shape^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
:*
swap_memory(
Ј
Ptraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/mul_5
ъ
Vtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations 
П
Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Const_1Const*%
_class
loc:@lstm_1/while/mul_5*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Const_1*

stack_name *
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/mul_5
Ы
Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc_1*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations 
ч
Straining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/mul_5_grad/Shape_1^training/Adam/gradients/Add*
_output_shapes
:*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/mul_5
Ќ
Rtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/mul_5
ю
Xtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterMtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc_1*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations 
ѓ
3training/Adam/gradients/lstm_1/while/mul_5_grad/MulMultraining/Adam/gradients/AddN_4>training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_5*'
_output_shapes
:џџџџџџџџџ
Х
9training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/ConstConst*?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_5*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
џ
9training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/f_accStackV29training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/Const*
_output_shapes
:*
	elem_type0*?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_5*

stack_name 
Ѓ
9training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/EnterEnter9training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
Ј
?training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPushV2StackPushV29training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/Enterlstm_1/while/Tanh_1^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_5*'
_output_shapes
:џџџџџџџџџ*
swap_memory(

>training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV2
StackPopV2Dtraining/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*
	elem_type0*%
_class
loc:@lstm_1/while/mul_5*'
_output_shapes
:џџџџџџџџџ
Ц
Dtraining/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV2/EnterEnter9training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations 

3training/Adam/gradients/lstm_1/while/mul_5_grad/SumSum3training/Adam/gradients/lstm_1/while/mul_5_grad/MulEtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs*
T0*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
:*
	keep_dims( *

Tidx0
А
7training/Adam/gradients/lstm_1/while/mul_5_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/mul_5_grad/SumPtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/mul_5*
Tshape0
ї
5training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1Mul@training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPopV2training/Adam/gradients/AddN_4*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/mul_5
а
;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/ConstConst*H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_5*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/f_accStackV2;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/Const*
_output_shapes
:*
	elem_type0*H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_5*

stack_name 
Ї
;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/EnterEnter;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5
Е
Atraining/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPushV2StackPushV2;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/Enterlstm_1/while/clip_by_value_2^training/Adam/gradients/Add*'
_output_shapes
:џџџџџџџџџ*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/mul_5

@training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPopV2
StackPopV2Ftraining/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPopV2/Enter^training/Adam/gradients/Sub*'
_output_shapes
:џџџџџџџџџ*
	elem_type0*%
_class
loc:@lstm_1/while/mul_5
Ъ
Ftraining/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPopV2/EnterEnter;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations 
Ѓ
5training/Adam/gradients/lstm_1/while/mul_5_grad/Sum_1Sum5training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1Gtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ж
9training/Adam/gradients/lstm_1/while/mul_5_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/mul_5_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2_1*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/mul_5*
Tshape0
п
training/Adam/gradients/AddN_5AddN,training/Adam/gradients/lstm_1/Tile_grad/Sum.training/Adam/gradients/lstm_1/Tile_1_grad/Sum*
T0*
_class
loc:@lstm_1/Tile*
N*'
_output_shapes
:џџџџџџџџџ
Є
4training/Adam/gradients/lstm_1/ExpandDims_grad/ShapeShape
lstm_1/Sum*
T0*$
_class
loc:@lstm_1/ExpandDims*
out_type0*
_output_shapes
:
љ
6training/Adam/gradients/lstm_1/ExpandDims_grad/ReshapeReshapetraining/Adam/gradients/AddN_54training/Adam/gradients/lstm_1/ExpandDims_grad/Shape*
T0*$
_class
loc:@lstm_1/ExpandDims*
Tshape0*#
_output_shapes
:џџџџџџџџџ
д
?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/ShapeShape$lstm_1/while/clip_by_value_2/Minimum*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
out_type0*
_output_shapes
:
г
Atraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Shape_1Const^training/Adam/gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
valueB *
dtype0*
_output_shapes
: 
щ
Atraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Shape_2Shape7training/Adam/gradients/lstm_1/while/mul_5_grad/Reshape*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
out_type0*
_output_shapes
:
й
Etraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/zeros/ConstConst^training/Adam/gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
valueB
 *    *
dtype0*
_output_shapes
: 
Ц
?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/zerosFillAtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Shape_2Etraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/zeros/Const*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*

index_type0*'
_output_shapes
:џџџџџџџџџ
м
Ftraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqualGreaterEqualQtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopV2Ntraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/Const_1*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*'
_output_shapes
:џџџџџџџџџ
ѓ
Ltraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/ConstConst*Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Р
Ltraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_accStackV2Ltraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/Const*Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*

stack_name *
_output_shapes
:*
	elem_type0
г
Ltraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/EnterEnterLtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
parallel_iterations 
щ
Rtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPushV2StackPushV2Ltraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/Enter$lstm_1/while/clip_by_value_2/Minimum^training/Adam/gradients/Add*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*'
_output_shapes
:џџџџџџџџџ*
swap_memory(
С
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopV2
StackPopV2Wtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopV2/Enter^training/Adam/gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
і
Wtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopV2/EnterEnterLtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
parallel_iterations 
т
Ntraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/Const_1Const^training/Adam/gradients/Sub*
_output_shapes
: */
_class%
#!loc:@lstm_1/while/clip_by_value_2*
valueB
 *    *
dtype0
ѕ
Otraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgsBroadcastGradientArgsZtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopV2Atraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2
б
Utraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/ConstConst*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ї
Utraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_accStackV2Utraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/Const*
	elem_type0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*

stack_name *
_output_shapes
:
х
Utraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/EnterEnterUtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_acc*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context

[training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2Utraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/Enter?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Shape^training/Adam/gradients/Add*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
_output_shapes
:*
swap_memory(
Ц
Ztraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2`training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
_output_shapes
:*
	elem_type0

`training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterUtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
parallel_iterations 
я
@training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/SelectSelectFtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual7training/Adam/gradients/lstm_1/while/mul_5_grad/Reshape?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/zeros*'
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2
ё
Btraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Select_1SelectFtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/zeros7training/Adam/gradients/lstm_1/while/mul_5_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2
Ш
=training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/SumSum@training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/SelectOtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2
и
Atraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/ReshapeReshape=training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/SumZtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
Tshape0
Ю
?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Sum_1SumBtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Select_1Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
_output_shapes
:
В
Ctraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Reshape_1Reshape?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Sum_1Atraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Shape_1*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
Tshape0*
_output_shapes
: 

9training/Adam/gradients/lstm_1/while/Tanh_1_grad/TanhGradTanhGrad>training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV29training/Adam/gradients/lstm_1/while/mul_5_grad/Reshape_1*
T0*&
_class
loc:@lstm_1/while/Tanh_1*'
_output_shapes
:џџџџџџџџџ
й
Btraining/Adam/gradients/lstm_1/while/Switch_2_grad_1/NextIterationNextIteration:training/Adam/gradients/lstm_1/while/Merge_2_grad/Switch:1*
T0*'
_class
loc:@lstm_1/while/Merge_2*
_output_shapes
: 

-training/Adam/gradients/lstm_1/Sum_grad/ShapeShapelstm_1/zeros_like*
T0*
_class
loc:@lstm_1/Sum*
out_type0*
_output_shapes
:

,training/Adam/gradients/lstm_1/Sum_grad/SizeConst*
_class
loc:@lstm_1/Sum*
value	B :*
dtype0*
_output_shapes
: 
Т
+training/Adam/gradients/lstm_1/Sum_grad/addAddlstm_1/Sum/reduction_indices,training/Adam/gradients/lstm_1/Sum_grad/Size*
T0*
_class
loc:@lstm_1/Sum*
_output_shapes
:
ж
+training/Adam/gradients/lstm_1/Sum_grad/modFloorMod+training/Adam/gradients/lstm_1/Sum_grad/add,training/Adam/gradients/lstm_1/Sum_grad/Size*
T0*
_class
loc:@lstm_1/Sum*
_output_shapes
:

/training/Adam/gradients/lstm_1/Sum_grad/Shape_1Const*
_output_shapes
:*
_class
loc:@lstm_1/Sum*
valueB:*
dtype0

3training/Adam/gradients/lstm_1/Sum_grad/range/startConst*
_class
loc:@lstm_1/Sum*
value	B : *
dtype0*
_output_shapes
: 

3training/Adam/gradients/lstm_1/Sum_grad/range/deltaConst*
_class
loc:@lstm_1/Sum*
value	B :*
dtype0*
_output_shapes
: 

-training/Adam/gradients/lstm_1/Sum_grad/rangeRange3training/Adam/gradients/lstm_1/Sum_grad/range/start,training/Adam/gradients/lstm_1/Sum_grad/Size3training/Adam/gradients/lstm_1/Sum_grad/range/delta*
_class
loc:@lstm_1/Sum*
_output_shapes
:*

Tidx0

2training/Adam/gradients/lstm_1/Sum_grad/Fill/valueConst*
_class
loc:@lstm_1/Sum*
value	B :*
dtype0*
_output_shapes
: 
я
,training/Adam/gradients/lstm_1/Sum_grad/FillFill/training/Adam/gradients/lstm_1/Sum_grad/Shape_12training/Adam/gradients/lstm_1/Sum_grad/Fill/value*
T0*
_class
loc:@lstm_1/Sum*

index_type0*
_output_shapes
:
Ь
5training/Adam/gradients/lstm_1/Sum_grad/DynamicStitchDynamicStitch-training/Adam/gradients/lstm_1/Sum_grad/range+training/Adam/gradients/lstm_1/Sum_grad/mod-training/Adam/gradients/lstm_1/Sum_grad/Shape,training/Adam/gradients/lstm_1/Sum_grad/Fill*
T0*
_class
loc:@lstm_1/Sum*
N*
_output_shapes
:

1training/Adam/gradients/lstm_1/Sum_grad/Maximum/yConst*
_class
loc:@lstm_1/Sum*
value	B :*
dtype0*
_output_shapes
: 
ш
/training/Adam/gradients/lstm_1/Sum_grad/MaximumMaximum5training/Adam/gradients/lstm_1/Sum_grad/DynamicStitch1training/Adam/gradients/lstm_1/Sum_grad/Maximum/y*
T0*
_class
loc:@lstm_1/Sum*
_output_shapes
:
р
0training/Adam/gradients/lstm_1/Sum_grad/floordivFloorDiv-training/Adam/gradients/lstm_1/Sum_grad/Shape/training/Adam/gradients/lstm_1/Sum_grad/Maximum*
_output_shapes
:*
T0*
_class
loc:@lstm_1/Sum

/training/Adam/gradients/lstm_1/Sum_grad/ReshapeReshape6training/Adam/gradients/lstm_1/ExpandDims_grad/Reshape5training/Adam/gradients/lstm_1/Sum_grad/DynamicStitch*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*
_class
loc:@lstm_1/Sum*
Tshape0
џ
,training/Adam/gradients/lstm_1/Sum_grad/TileTile/training/Adam/gradients/lstm_1/Sum_grad/Reshape0training/Adam/gradients/lstm_1/Sum_grad/floordiv*
_class
loc:@lstm_1/Sum*,
_output_shapes
:џџџџџџџџџЅ@*

Tmultiples0*
T0
в
Gtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ShapeShapelstm_1/while/add_8*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
out_type0*
_output_shapes
:
у
Itraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_1Const^training/Adam/gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB *
dtype0*
_output_shapes
: 

Itraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_2ShapeAtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Reshape*
_output_shapes
:*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
out_type0
щ
Mtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros/ConstConst^training/Adam/gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB
 *    *
dtype0*
_output_shapes
: 
ц
Gtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zerosFillItraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_2Mtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros/Const*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*

index_type0*'
_output_shapes
:џџџџџџџџџ
№
Ktraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual	LessEqualVtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopV2Straining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/Const_1*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*'
_output_shapes
:џџџџџџџџџ
ю
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/ConstConst*P
_classF
Dloc:@lstm_1/while/add_8)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Р
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_accStackV2Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/Const*P
_classF
Dloc:@lstm_1/while/add_8)loc:@lstm_1/while/clip_by_value_2/Minimum*

stack_name *
_output_shapes
:*
	elem_type0
х
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/EnterEnterQtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
parallel_iterations 
щ
Wtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPushV2StackPushV2Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/Enterlstm_1/while/add_8^training/Adam/gradients/Add*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*'
_output_shapes
:џџџџџџџџџ*
swap_memory(*
T0
г
Vtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopV2
StackPopV2\training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopV2/Enter^training/Adam/gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*'
_output_shapes
:џџџџџџџџџ*
	elem_type0

\training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopV2/EnterEnterQtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
parallel_iterations 
я
Straining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/Const_1Const^training/Adam/gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Wtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsbtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopV2Itraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_1*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
с
]training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/ConstConst*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
П
]training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_accStackV2]training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/Const*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*

stack_name *
_output_shapes
:*
	elem_type0
§
]training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/EnterEnter]training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
parallel_iterations 
Љ
ctraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPushV2StackPushV2]training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/EnterGtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape^training/Adam/gradients/Add*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
_output_shapes
:*
swap_memory(
о
btraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopV2
StackPopV2htraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
	elem_type0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
_output_shapes
:
 
htraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopV2/EnterEnter]training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
parallel_iterations 

Htraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SelectSelectKtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqualAtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/ReshapeGtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*'
_output_shapes
:џџџџџџџџџ

Jtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Select_1SelectKtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqualGtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zerosAtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Reshape*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*'
_output_shapes
:џџџџџџџџџ*
T0
ш
Etraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SumSumHtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SelectWtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum
ј
Itraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ReshapeReshapeEtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Sumbtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
Tshape0
ю
Gtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Sum_1SumJtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Select_1Ytraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
_output_shapes
:
в
Ktraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Reshape_1ReshapeGtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Sum_1Itraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_1*
_output_shapes
: *
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
Tshape0

training/Adam/gradients/AddN_6AddN:training/Adam/gradients/lstm_1/while/Merge_4_grad/Switch:19training/Adam/gradients/lstm_1/while/Tanh_1_grad/TanhGrad*'
_class
loc:@lstm_1/while/Merge_4*
N*'
_output_shapes
:џџџџџџџџџ*
T0
Ў
5training/Adam/gradients/lstm_1/while/add_6_grad/ShapeShapelstm_1/while/mul_2*
T0*%
_class
loc:@lstm_1/while/add_6*
out_type0*
_output_shapes
:
А
7training/Adam/gradients/lstm_1/while/add_6_grad/Shape_1Shapelstm_1/while/mul_3*
_output_shapes
:*
T0*%
_class
loc:@lstm_1/while/add_6*
out_type0
ш
Etraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2Rtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/add_6*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/add_6*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/add_6*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_6*
parallel_iterations 
с
Qtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/add_6_grad/Shape^training/Adam/gradients/Add*%
_class
loc:@lstm_1/while/add_6*
_output_shapes
:*
swap_memory(*
T0
Ј
Ptraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_6*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_6*
parallel_iterations 
П
Mtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/Const_1Const*%
_class
loc:@lstm_1/while/add_6*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Mtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/Const_1*%
_class
loc:@lstm_1/while/add_6*

stack_name *
_output_shapes
:*
	elem_type0
Ы
Mtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1*
T0*%
_class
loc:@lstm_1/while/add_6*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
ч
Straining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/add_6_grad/Shape_1^training/Adam/gradients/Add*
_output_shapes
:*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/add_6
Ќ
Rtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_6*
_output_shapes
:*
	elem_type0
ю
Xtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterMtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_6*
parallel_iterations *
is_constant(

3training/Adam/gradients/lstm_1/while/add_6_grad/SumSumtraining/Adam/gradients/AddN_6Etraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_6*
_output_shapes
:
А
7training/Adam/gradients/lstm_1/while/add_6_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_6_grad/SumPtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/add_6*
Tshape0*'
_output_shapes
:џџџџџџџџџ

5training/Adam/gradients/lstm_1/while/add_6_grad/Sum_1Sumtraining/Adam/gradients/AddN_6Gtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/add_6*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ж
9training/Adam/gradients/lstm_1/while/add_6_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_6_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/add_6*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ў
5training/Adam/gradients/lstm_1/while/add_8_grad/ShapeShapelstm_1/while/mul_4*
_output_shapes
:*
T0*%
_class
loc:@lstm_1/while/add_8*
out_type0
П
7training/Adam/gradients/lstm_1/while/add_8_grad/Shape_1Const^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_8*
valueB *
dtype0*
_output_shapes
: 
Э
Etraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPopV27training/Adam/gradients/lstm_1/while/add_8_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/add_8
Н
Ktraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/add_8*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/Const*
	elem_type0*%
_class
loc:@lstm_1/while/add_8*

stack_name *
_output_shapes
:
Ч
Ktraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_8*
parallel_iterations *
is_constant(
с
Qtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/add_8_grad/Shape^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/add_8*
_output_shapes
:*
swap_memory(
Ј
Ptraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
	elem_type0*%
_class
loc:@lstm_1/while/add_8*
_output_shapes
:
ъ
Vtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_8*
parallel_iterations 
Г
3training/Adam/gradients/lstm_1/while/add_8_grad/SumSumItraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ReshapeEtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs*
T0*%
_class
loc:@lstm_1/while/add_8*
_output_shapes
:*
	keep_dims( *

Tidx0
А
7training/Adam/gradients/lstm_1/while/add_8_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_8_grad/SumPtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/add_8*
Tshape0*'
_output_shapes
:џџџџџџџџџ
З
5training/Adam/gradients/lstm_1/while/add_8_grad/Sum_1SumItraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ReshapeGtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@lstm_1/while/add_8*
_output_shapes
:*
	keep_dims( *

Tidx0

9training/Adam/gradients/lstm_1/while/add_8_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_8_grad/Sum_17training/Adam/gradients/lstm_1/while/add_8_grad/Shape_1*
T0*%
_class
loc:@lstm_1/while/add_8*
Tshape0*
_output_shapes
: 
И
5training/Adam/gradients/lstm_1/while/mul_2_grad/ShapeShapelstm_1/while/clip_by_value_1*
T0*%
_class
loc:@lstm_1/while/mul_2*
out_type0*
_output_shapes
:
Е
7training/Adam/gradients/lstm_1/while/mul_2_grad/Shape_1Shapelstm_1/while/Identity_4*
T0*%
_class
loc:@lstm_1/while/mul_2*
out_type0*
_output_shapes
:
ш
Etraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2Rtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/mul_2*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/mul_2*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/mul_2*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations 
с
Qtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/mul_2_grad/Shape^training/Adam/gradients/Add*
_output_shapes
:*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/mul_2
Ј
Ptraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
	elem_type0*%
_class
loc:@lstm_1/while/mul_2*
_output_shapes
:
ъ
Vtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0
П
Mtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *%
_class
loc:@lstm_1/while/mul_2*
valueB :
џџџџџџџџџ

Mtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/Const_1*%
_class
loc:@lstm_1/while/mul_2*

stack_name *
_output_shapes
:*
	elem_type0
Ы
Mtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
ч
Straining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/mul_2_grad/Shape_1^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_2*
_output_shapes
:*
swap_memory(
Ќ
Rtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_2*
_output_shapes
:*
	elem_type0
ю
Xtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterMtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations *
is_constant(

3training/Adam/gradients/lstm_1/while/mul_2_grad/MulMul7training/Adam/gradients/lstm_1/while/add_6_grad/Reshape>training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPopV2*%
_class
loc:@lstm_1/while/mul_2*'
_output_shapes
:џџџџџџџџџ*
T0
Щ
9training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/ConstConst*C
_class9
7loc:@lstm_1/while/Identity_4loc:@lstm_1/while/mul_2*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

9training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/f_accStackV29training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/Const*C
_class9
7loc:@lstm_1/while/Identity_4loc:@lstm_1/while/mul_2*

stack_name *
_output_shapes
:*
	elem_type0
Ѓ
9training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/EnterEnter9training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations 
Ќ
?training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPushV2StackPushV29training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/Enterlstm_1/while/Identity_4^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_2*'
_output_shapes
:џџџџџџџџџ*
swap_memory(

>training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPopV2
StackPopV2Dtraining/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*'
_output_shapes
:џџџџџџџџџ*
	elem_type0*%
_class
loc:@lstm_1/while/mul_2
Ц
Dtraining/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPopV2/EnterEnter9training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context

3training/Adam/gradients/lstm_1/while/mul_2_grad/SumSum3training/Adam/gradients/lstm_1/while/mul_2_grad/MulEtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/mul_2*
_output_shapes
:
А
7training/Adam/gradients/lstm_1/while/mul_2_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/mul_2_grad/SumPtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_2*
Tshape0*'
_output_shapes
:џџџџџџџџџ

5training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1Mul@training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPopV27training/Adam/gradients/lstm_1/while/add_6_grad/Reshape*
T0*%
_class
loc:@lstm_1/while/mul_2*'
_output_shapes
:џџџџџџџџџ
а
;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/ConstConst*H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_2*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/f_accStackV2;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/Const*H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_2*

stack_name *
_output_shapes
:*
	elem_type0
Ї
;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/EnterEnter;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations *
is_constant(
Е
Atraining/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPushV2StackPushV2;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/Enterlstm_1/while/clip_by_value_1^training/Adam/gradients/Add*'
_output_shapes
:џџџџџџџџџ*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/mul_2

@training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPopV2
StackPopV2Ftraining/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_2*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
Ъ
Ftraining/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPopV2/EnterEnter;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
Ѓ
5training/Adam/gradients/lstm_1/while/mul_2_grad/Sum_1Sum5training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1Gtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/mul_2
Ж
9training/Adam/gradients/lstm_1/while/mul_2_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/mul_2_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/mul_2*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ж
5training/Adam/gradients/lstm_1/while/mul_3_grad/ShapeShapelstm_1/while/clip_by_value*%
_class
loc:@lstm_1/while/mul_3*
out_type0*
_output_shapes
:*
T0
Џ
7training/Adam/gradients/lstm_1/while/mul_3_grad/Shape_1Shapelstm_1/while/Tanh*
T0*%
_class
loc:@lstm_1/while/mul_3*
out_type0*
_output_shapes
:
ш
Etraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2Rtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2_1*%
_class
loc:@lstm_1/while/mul_3*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Н
Ktraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/mul_3*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Const*
	elem_type0*%
_class
loc:@lstm_1/while/mul_3*

stack_name *
_output_shapes
:
Ч
Ktraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations *
is_constant(
с
Qtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/mul_3_grad/Shape^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_3*
_output_shapes
:*
swap_memory(
Ј
Ptraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_3*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations *
is_constant(
П
Mtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Const_1Const*%
_class
loc:@lstm_1/while/mul_3*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Mtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Const_1*%
_class
loc:@lstm_1/while/mul_3*

stack_name *
_output_shapes
:*
	elem_type0
Ы
Mtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc_1*
T0*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
ч
Straining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/mul_3_grad/Shape_1^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_3*
_output_shapes
:*
swap_memory(
Ќ
Rtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/mul_3
ю
Xtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterMtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc_1*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations 

3training/Adam/gradients/lstm_1/while/mul_3_grad/MulMul9training/Adam/gradients/lstm_1/while/add_6_grad/Reshape_1>training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_3*'
_output_shapes
:џџџџџџџџџ
У
9training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/ConstConst*
_output_shapes
: *=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_3*
valueB :
џџџџџџџџџ*
dtype0
§
9training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/f_accStackV29training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/Const*=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_3*

stack_name *
_output_shapes
:*
	elem_type0
Ѓ
9training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/EnterEnter9training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/f_acc*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0
І
?training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPushV2StackPushV29training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/Enterlstm_1/while/Tanh^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_3*'
_output_shapes
:џџџџџџџџџ*
swap_memory(

>training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPopV2
StackPopV2Dtraining/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_3*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
Ц
Dtraining/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPopV2/EnterEnter9training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations 

3training/Adam/gradients/lstm_1/while/mul_3_grad/SumSum3training/Adam/gradients/lstm_1/while/mul_3_grad/MulEtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/mul_3*
_output_shapes
:
А
7training/Adam/gradients/lstm_1/while/mul_3_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/mul_3_grad/SumPtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2*%
_class
loc:@lstm_1/while/mul_3*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0

5training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1Mul@training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPopV29training/Adam/gradients/lstm_1/while/add_6_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/mul_3
Ю
;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_3*
valueB :
џџџџџџџџџ

;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/f_accStackV2;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/Const*
_output_shapes
:*
	elem_type0*F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_3*

stack_name 
Ї
;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/EnterEnter;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
Г
Atraining/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPushV2StackPushV2;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/Enterlstm_1/while/clip_by_value^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_3*'
_output_shapes
:џџџџџџџџџ*
swap_memory(

@training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPopV2
StackPopV2Ftraining/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_3*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
Ъ
Ftraining/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPopV2/EnterEnter;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations 
Ѓ
5training/Adam/gradients/lstm_1/while/mul_3_grad/Sum_1Sum5training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1Gtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@lstm_1/while/mul_3*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
9training/Adam/gradients/lstm_1/while/mul_3_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/mul_3_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/mul_3*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Н
5training/Adam/gradients/lstm_1/while/mul_4_grad/ShapeConst^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_4*
valueB *
dtype0*
_output_shapes
: 
А
7training/Adam/gradients/lstm_1/while/mul_4_grad/Shape_1Shapelstm_1/while/add_7*
T0*%
_class
loc:@lstm_1/while/mul_4*
out_type0*
_output_shapes
:
Ы
Etraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/lstm_1/while/mul_4_grad/ShapePtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopV2*%
_class
loc:@lstm_1/while/mul_4*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Н
Ktraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/mul_4*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/mul_4*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc*%
_class
loc:@lstm_1/while/mul_4*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0
у
Qtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/Enter7training/Adam/gradients/lstm_1/while/mul_4_grad/Shape_1^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_4*
_output_shapes
:*
swap_memory(
Ј
Ptraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/mul_4
ъ
Vtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc*%
_class
loc:@lstm_1/while/mul_4*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0

3training/Adam/gradients/lstm_1/while/mul_4_grad/MulMul7training/Adam/gradients/lstm_1/while/add_8_grad/Reshape>training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_4*'
_output_shapes
:џџџџџџџџџ
Ф
9training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/ConstConst*>
_class4
2loc:@lstm_1/while/add_7loc:@lstm_1/while/mul_4*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
ў
9training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/f_accStackV29training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/Const*>
_class4
2loc:@lstm_1/while/add_7loc:@lstm_1/while/mul_4*

stack_name *
_output_shapes
:*
	elem_type0
Ѓ
9training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/EnterEnter9training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_4*
parallel_iterations 
Ї
?training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPushV2StackPushV29training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/Enterlstm_1/while/add_7^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_4*'
_output_shapes
:џџџџџџџџџ*
swap_memory(

>training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPopV2
StackPopV2Dtraining/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*'
_output_shapes
:џџџџџџџџџ*
	elem_type0*%
_class
loc:@lstm_1/while/mul_4
Ц
Dtraining/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPopV2/EnterEnter9training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/f_acc*%
_class
loc:@lstm_1/while/mul_4*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0

3training/Adam/gradients/lstm_1/while/mul_4_grad/SumSum3training/Adam/gradients/lstm_1/while/mul_4_grad/MulEtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/mul_4*
_output_shapes
:

7training/Adam/gradients/lstm_1/while/mul_4_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/mul_4_grad/Sum5training/Adam/gradients/lstm_1/while/mul_4_grad/Shape*
T0*%
_class
loc:@lstm_1/while/mul_4*
Tshape0*
_output_shapes
: 

5training/Adam/gradients/lstm_1/while/mul_4_grad/Mul_1Mul;training/Adam/gradients/lstm_1/while/mul_4_grad/Mul_1/Const7training/Adam/gradients/lstm_1/while/add_8_grad/Reshape*
T0*%
_class
loc:@lstm_1/while/mul_4*'
_output_shapes
:џџџџџџџџџ
Х
;training/Adam/gradients/lstm_1/while/mul_4_grad/Mul_1/ConstConst^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_4*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
Ѓ
5training/Adam/gradients/lstm_1/while/mul_4_grad/Sum_1Sum5training/Adam/gradients/lstm_1/while/mul_4_grad/Mul_1Gtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/mul_4
Д
9training/Adam/gradients/lstm_1/while/mul_4_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/mul_4_grad/Sum_1Ptraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_4*
Tshape0*'
_output_shapes
:џџџџџџџџџ
д
?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/ShapeShape$lstm_1/while/clip_by_value_1/Minimum*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
out_type0*
_output_shapes
:
г
Atraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Shape_1Const^training/Adam/gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
valueB *
dtype0*
_output_shapes
: 
щ
Atraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Shape_2Shape7training/Adam/gradients/lstm_1/while/mul_2_grad/Reshape*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
out_type0*
_output_shapes
:
й
Etraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/zeros/ConstConst^training/Adam/gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
valueB
 *    *
dtype0*
_output_shapes
: 
Ц
?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/zerosFillAtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Shape_2Etraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/zeros/Const*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*

index_type0*'
_output_shapes
:џџџџџџџџџ
м
Ftraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqualGreaterEqualQtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopV2Ntraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/Const_1*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*'
_output_shapes
:џџџџџџџџџ
ѓ
Ltraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/ConstConst*Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Р
Ltraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_accStackV2Ltraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/Const*Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*

stack_name *
_output_shapes
:*
	elem_type0
г
Ltraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/EnterEnterLtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
parallel_iterations 
щ
Rtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPushV2StackPushV2Ltraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/Enter$lstm_1/while/clip_by_value_1/Minimum^training/Adam/gradients/Add*'
_output_shapes
:џџџџџџџџџ*
swap_memory(*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1
С
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopV2
StackPopV2Wtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopV2/Enter^training/Adam/gradients/Sub*
	elem_type0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*'
_output_shapes
:џџџџџџџџџ
і
Wtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopV2/EnterEnterLtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
т
Ntraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/Const_1Const^training/Adam/gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
valueB
 *    *
dtype0*
_output_shapes
: 
ѕ
Otraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgsBroadcastGradientArgsZtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopV2Atraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Shape_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
б
Utraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/ConstConst*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ї
Utraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_accStackV2Utraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/Const*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*

stack_name *
_output_shapes
:*
	elem_type0
х
Utraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/EnterEnterUtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_acc*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context

[training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Utraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/Enter?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Shape^training/Adam/gradients/Add*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
_output_shapes
:*
swap_memory(
Ц
Ztraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2`training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
_output_shapes
:*
	elem_type0

`training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterUtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1
я
@training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/SelectSelectFtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual7training/Adam/gradients/lstm_1/while/mul_2_grad/Reshape?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/zeros*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*'
_output_shapes
:џџџџџџџџџ
ё
Btraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Select_1SelectFtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/zeros7training/Adam/gradients/lstm_1/while/mul_2_grad/Reshape*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*'
_output_shapes
:џџџџџџџџџ
Ш
=training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/SumSum@training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/SelectOtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
_output_shapes
:
и
Atraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/ReshapeReshape=training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/SumZtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
Tshape0
Ю
?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Sum_1SumBtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Select_1Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1
В
Ctraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Reshape_1Reshape?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Sum_1Atraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Shape_1*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
Tshape0*
_output_shapes
: 
Ю
=training/Adam/gradients/lstm_1/while/clip_by_value_grad/ShapeShape"lstm_1/while/clip_by_value/Minimum*
_output_shapes
:*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
out_type0
Я
?training/Adam/gradients/lstm_1/while/clip_by_value_grad/Shape_1Const^training/Adam/gradients/Sub*-
_class#
!loc:@lstm_1/while/clip_by_value*
valueB *
dtype0*
_output_shapes
: 
х
?training/Adam/gradients/lstm_1/while/clip_by_value_grad/Shape_2Shape7training/Adam/gradients/lstm_1/while/mul_3_grad/Reshape*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
out_type0*
_output_shapes
:
е
Ctraining/Adam/gradients/lstm_1/while/clip_by_value_grad/zeros/ConstConst^training/Adam/gradients/Sub*-
_class#
!loc:@lstm_1/while/clip_by_value*
valueB
 *    *
dtype0*
_output_shapes
: 
О
=training/Adam/gradients/lstm_1/while/clip_by_value_grad/zerosFill?training/Adam/gradients/lstm_1/while/clip_by_value_grad/Shape_2Ctraining/Adam/gradients/lstm_1/while/clip_by_value_grad/zeros/Const*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*

index_type0*'
_output_shapes
:џџџџџџџџџ
д
Dtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqualGreaterEqualOtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopV2Ltraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/Const_1*-
_class#
!loc:@lstm_1/while/clip_by_value*'
_output_shapes
:џџџџџџџџџ*
T0
э
Jtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/ConstConst*
_output_shapes
: *V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*
valueB :
џџџџџџџџџ*
dtype0
И
Jtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_accStackV2Jtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/Const*V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*

stack_name *
_output_shapes
:*
	elem_type0
Э
Jtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/EnterEnterJtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value
с
Ptraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPushV2StackPushV2Jtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/Enter"lstm_1/while/clip_by_value/Minimum^training/Adam/gradients/Add*'
_output_shapes
:џџџџџџџџџ*
swap_memory(*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value
Л
Otraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopV2
StackPopV2Utraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopV2/Enter^training/Adam/gradients/Sub*'
_output_shapes
:џџџџџџџџџ*
	elem_type0*-
_class#
!loc:@lstm_1/while/clip_by_value
№
Utraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopV2/EnterEnterJtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
parallel_iterations *
is_constant(
о
Ltraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/Const_1Const^training/Adam/gradients/Sub*-
_class#
!loc:@lstm_1/while/clip_by_value*
valueB
 *    *
dtype0*
_output_shapes
: 
э
Mtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsXtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopV2?training/Adam/gradients/lstm_1/while/clip_by_value_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value
Э
Straining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/ConstConst*-
_class#
!loc:@lstm_1/while/clip_by_value*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ё
Straining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_accStackV2Straining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/Const*-
_class#
!loc:@lstm_1/while/clip_by_value*

stack_name *
_output_shapes
:*
	elem_type0
п
Straining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/EnterEnterStraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
parallel_iterations *
is_constant(

Ytraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPushV2StackPushV2Straining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/Enter=training/Adam/gradients/lstm_1/while/clip_by_value_grad/Shape^training/Adam/gradients/Add*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
_output_shapes
:*
swap_memory(
Р
Xtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopV2
StackPopV2^training/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*-
_class#
!loc:@lstm_1/while/clip_by_value*
_output_shapes
:*
	elem_type0

^training/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopV2/EnterEnterStraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
parallel_iterations 
ч
>training/Adam/gradients/lstm_1/while/clip_by_value_grad/SelectSelectDtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual7training/Adam/gradients/lstm_1/while/mul_3_grad/Reshape=training/Adam/gradients/lstm_1/while/clip_by_value_grad/zeros*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*'
_output_shapes
:џџџџџџџџџ
щ
@training/Adam/gradients/lstm_1/while/clip_by_value_grad/Select_1SelectDtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual=training/Adam/gradients/lstm_1/while/clip_by_value_grad/zeros7training/Adam/gradients/lstm_1/while/mul_3_grad/Reshape*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*'
_output_shapes
:џџџџџџџџџ
Р
;training/Adam/gradients/lstm_1/while/clip_by_value_grad/SumSum>training/Adam/gradients/lstm_1/while/clip_by_value_grad/SelectMtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs*-
_class#
!loc:@lstm_1/while/clip_by_value*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
а
?training/Adam/gradients/lstm_1/while/clip_by_value_grad/ReshapeReshape;training/Adam/gradients/lstm_1/while/clip_by_value_grad/SumXtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopV2*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ц
=training/Adam/gradients/lstm_1/while/clip_by_value_grad/Sum_1Sum@training/Adam/gradients/lstm_1/while/clip_by_value_grad/Select_1Otraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
_output_shapes
:
Њ
Atraining/Adam/gradients/lstm_1/while/clip_by_value_grad/Reshape_1Reshape=training/Adam/gradients/lstm_1/while/clip_by_value_grad/Sum_1?training/Adam/gradients/lstm_1/while/clip_by_value_grad/Shape_1*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
Tshape0*
_output_shapes
: 

7training/Adam/gradients/lstm_1/while/Tanh_grad/TanhGradTanhGrad>training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPopV29training/Adam/gradients/lstm_1/while/mul_3_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ*
T0*$
_class
loc:@lstm_1/while/Tanh
В
5training/Adam/gradients/lstm_1/while/add_7_grad/ShapeShapelstm_1/while/BiasAdd_3*
T0*%
_class
loc:@lstm_1/while/add_7*
out_type0*
_output_shapes
:
Г
7training/Adam/gradients/lstm_1/while/add_7_grad/Shape_1Shapelstm_1/while/MatMul_7*
T0*%
_class
loc:@lstm_1/while/add_7*
out_type0*
_output_shapes
:
ш
Etraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2Rtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/add_7*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/add_7*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/add_7*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_7*
parallel_iterations *
is_constant(
с
Qtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/add_7_grad/Shape^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/add_7*
_output_shapes
:*
swap_memory(
Ј
Ptraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_7*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_7
П
Mtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *%
_class
loc:@lstm_1/while/add_7*
valueB :
џџџџџџџџџ

Mtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/Const_1*%
_class
loc:@lstm_1/while/add_7*

stack_name *
_output_shapes
:*
	elem_type0
Ы
Mtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc_1*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_7*
parallel_iterations 
ч
Straining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/add_7_grad/Shape_1^training/Adam/gradients/Add*
_output_shapes
:*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/add_7
Ќ
Rtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/add_7
ю
Xtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterMtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc_1*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_7*
parallel_iterations 
Ѓ
3training/Adam/gradients/lstm_1/while/add_7_grad/SumSum9training/Adam/gradients/lstm_1/while/mul_4_grad/Reshape_1Etraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs*
T0*%
_class
loc:@lstm_1/while/add_7*
_output_shapes
:*
	keep_dims( *

Tidx0
А
7training/Adam/gradients/lstm_1/while/add_7_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_7_grad/SumPtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/add_7*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ї
5training/Adam/gradients/lstm_1/while/add_7_grad/Sum_1Sum9training/Adam/gradients/lstm_1/while/mul_4_grad/Reshape_1Gtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@lstm_1/while/add_7*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
9training/Adam/gradients/lstm_1/while/add_7_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_7_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2_1*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/add_7*
Tshape0
в
Gtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/ShapeShapelstm_1/while/add_4*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
out_type0*
_output_shapes
:*
T0
у
Itraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_1Const^training/Adam/gradients/Sub*
_output_shapes
: *7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
valueB *
dtype0

Itraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_2ShapeAtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Reshape*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
out_type0*
_output_shapes
:
щ
Mtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros/ConstConst^training/Adam/gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
valueB
 *    *
dtype0*
_output_shapes
: 
ц
Gtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zerosFillItraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_2Mtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros/Const*'
_output_shapes
:џџџџџџџџџ*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*

index_type0
№
Ktraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual	LessEqualVtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopV2Straining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/Const_1*'
_output_shapes
:џџџџџџџџџ*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum
ю
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/ConstConst*P
_classF
Dloc:@lstm_1/while/add_4)loc:@lstm_1/while/clip_by_value_1/Minimum*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Р
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_accStackV2Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/Const*

stack_name *
_output_shapes
:*
	elem_type0*P
_classF
Dloc:@lstm_1/while/add_4)loc:@lstm_1/while/clip_by_value_1/Minimum
х
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/EnterEnterQtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
щ
Wtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPushV2StackPushV2Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/Enterlstm_1/while/add_4^training/Adam/gradients/Add*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*'
_output_shapes
:џџџџџџџџџ*
swap_memory(
г
Vtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopV2
StackPopV2\training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopV2/Enter^training/Adam/gradients/Sub*
	elem_type0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*'
_output_shapes
:џџџџџџџџџ

\training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopV2/EnterEnterQtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
parallel_iterations 
я
Straining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/Const_1Const^training/Adam/gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Wtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsbtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopV2Itraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_1*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
с
]training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/ConstConst*
_output_shapes
: *7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
valueB :
џџџџџџџџџ*
dtype0
П
]training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_accStackV2]training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/Const*
_output_shapes
:*
	elem_type0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*

stack_name 
§
]training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/EnterEnter]training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_acc*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
Љ
ctraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPushV2StackPushV2]training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/EnterGtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape^training/Adam/gradients/Add*
_output_shapes
:*
swap_memory(*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum
о
btraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopV2
StackPopV2htraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
_output_shapes
:*
	elem_type0
 
htraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopV2/EnterEnter]training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_acc*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context

Htraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SelectSelectKtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqualAtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/ReshapeGtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*'
_output_shapes
:џџџџџџџџџ*
T0

Jtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Select_1SelectKtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqualGtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zerosAtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Reshape*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*'
_output_shapes
:џџџџџџџџџ*
T0
ш
Etraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SumSumHtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SelectWtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
_output_shapes
:
ј
Itraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/ReshapeReshapeEtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Sumbtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopV2*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ю
Gtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Sum_1SumJtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Select_1Ytraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs:1*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
_output_shapes
:*
	keep_dims( *

Tidx0
в
Ktraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Reshape_1ReshapeGtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Sum_1Itraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_1*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
Tshape0*
_output_shapes
: 
щ
Btraining/Adam/gradients/lstm_1/while/Switch_4_grad_1/NextIterationNextIteration9training/Adam/gradients/lstm_1/while/mul_2_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ*
T0*'
_class
loc:@lstm_1/while/Merge_4
Ю
Etraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/ShapeShapelstm_1/while/add_2*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
out_type0*
_output_shapes
:
п
Gtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_1Const^training/Adam/gradients/Sub*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
valueB *
dtype0*
_output_shapes
: 
§
Gtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_2Shape?training/Adam/gradients/lstm_1/while/clip_by_value_grad/Reshape*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
out_type0*
_output_shapes
:
х
Ktraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros/ConstConst^training/Adam/gradients/Sub*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
valueB
 *    *
dtype0*
_output_shapes
: 
о
Etraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/zerosFillGtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_2Ktraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros/Const*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*

index_type0*'
_output_shapes
:џџџџџџџџџ
ш
Itraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual	LessEqualTtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopV2Qtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/Const_1*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ
ъ
Otraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/ConstConst*N
_classD
Bloc:@lstm_1/while/add_2'loc:@lstm_1/while/clip_by_value/Minimum*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
К
Otraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_accStackV2Otraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/Const*
_output_shapes
:*
	elem_type0*N
_classD
Bloc:@lstm_1/while/add_2'loc:@lstm_1/while/clip_by_value/Minimum*

stack_name 
п
Otraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/EnterEnterOtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum
у
Utraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPushV2StackPushV2Otraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/Enterlstm_1/while/add_2^training/Adam/gradients/Add*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ*
swap_memory(
Э
Ttraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopV2
StackPopV2Ztraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopV2/Enter^training/Adam/gradients/Sub*
	elem_type0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ

Ztraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopV2/EnterEnterOtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
parallel_iterations *
is_constant(
ы
Qtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/Const_1Const^training/Adam/gradients/Sub*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Utraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs`training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopV2Gtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_1*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
н
[training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/ConstConst*
_output_shapes
: *5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
valueB :
џџџџџџџџџ*
dtype0
Й
[training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_accStackV2[training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/Const*
	elem_type0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*

stack_name *
_output_shapes
:
ї
[training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/EnterEnter[training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
parallel_iterations 
Ё
atraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPushV2StackPushV2[training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/EnterEtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape^training/Adam/gradients/Add*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
_output_shapes
:*
swap_memory(
и
`training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ftraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
_output_shapes
:*
	elem_type0

ftraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopV2/EnterEnter[training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
parallel_iterations 

Ftraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/SelectSelectItraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual?training/Adam/gradients/lstm_1/while/clip_by_value_grad/ReshapeEtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ

Htraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Select_1SelectItraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqualEtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros?training/Adam/gradients/lstm_1/while/clip_by_value_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum
р
Ctraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/SumSumFtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/SelectUtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
№
Gtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/ReshapeReshapeCtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Sum`training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
Tshape0
ц
Etraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Sum_1SumHtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Select_1Wtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ъ
Itraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Reshape_1ReshapeEtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Sum_1Gtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_1*
_output_shapes
: *
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
Tshape0
В
5training/Adam/gradients/lstm_1/while/add_5_grad/ShapeShapelstm_1/while/BiasAdd_2*
T0*%
_class
loc:@lstm_1/while/add_5*
out_type0*
_output_shapes
:
Г
7training/Adam/gradients/lstm_1/while/add_5_grad/Shape_1Shapelstm_1/while/MatMul_6*
T0*%
_class
loc:@lstm_1/while/add_5*
out_type0*
_output_shapes
:
ш
Etraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2Rtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/add_5*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/add_5*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/add_5*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_5*
parallel_iterations 
с
Qtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/add_5_grad/Shape^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/add_5*
_output_shapes
:*
swap_memory(
Ј
Ptraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_5*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_5*
parallel_iterations 
П
Mtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/Const_1Const*%
_class
loc:@lstm_1/while/add_5*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Mtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/Const_1*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/add_5*

stack_name 
Ы
Mtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1*
T0*%
_class
loc:@lstm_1/while/add_5*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
ч
Straining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/add_5_grad/Shape_1^training/Adam/gradients/Add*
_output_shapes
:*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/add_5
Ќ
Rtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_5*
_output_shapes
:*
	elem_type0
ю
Xtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterMtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_5
Ё
3training/Adam/gradients/lstm_1/while/add_5_grad/SumSum7training/Adam/gradients/lstm_1/while/Tanh_grad/TanhGradEtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs*
T0*%
_class
loc:@lstm_1/while/add_5*
_output_shapes
:*
	keep_dims( *

Tidx0
А
7training/Adam/gradients/lstm_1/while/add_5_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_5_grad/SumPtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/add_5*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ѕ
5training/Adam/gradients/lstm_1/while/add_5_grad/Sum_1Sum7training/Adam/gradients/lstm_1/while/Tanh_grad/TanhGradGtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@lstm_1/while/add_5*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
9training/Adam/gradients/lstm_1/while/add_5_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_5_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/add_5*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ю
?training/Adam/gradients/lstm_1/while/BiasAdd_3_grad/BiasAddGradBiasAddGrad7training/Adam/gradients/lstm_1/while/add_7_grad/Reshape*)
_class
loc:@lstm_1/while/BiasAdd_3*
data_formatNHWC*
_output_shapes
:*
T0
С
9training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMulMatMul9training/Adam/gradients/lstm_1/while/add_7_grad/Reshape_1?training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul/Enter*(
_class
loc:@lstm_1/while/MatMul_7*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul/EnterEnterlstm_1/strided_slice_7*
is_constant(*
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_7*
parallel_iterations 
С
;training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV29training/Adam/gradients/lstm_1/while/add_7_grad/Reshape_1*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0*(
_class
loc:@lstm_1/while/MatMul_7
д
Atraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/ConstConst*F
_class<
:loc:@lstm_1/while/Identity_3loc:@lstm_1/while/MatMul_7*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Atraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/f_accStackV2Atraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/Const*F
_class<
:loc:@lstm_1/while/Identity_3loc:@lstm_1/while/MatMul_7*

stack_name *
_output_shapes
:*
	elem_type0
Ж
Atraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/EnterEnterAtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_7*
parallel_iterations *
is_constant(
П
Gtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPushV2StackPushV2Atraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/Enterlstm_1/while/Identity_3^training/Adam/gradients/Add*
T0*(
_class
loc:@lstm_1/while/MatMul_7*'
_output_shapes
:џџџџџџџџџ*
swap_memory(
Є
Ftraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV2
StackPopV2Ltraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV2/Enter^training/Adam/gradients/Sub*(
_class
loc:@lstm_1/while/MatMul_7*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
й
Ltraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV2/EnterEnterAtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/f_acc*
T0*(
_class
loc:@lstm_1/while/MatMul_7*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
Ў
5training/Adam/gradients/lstm_1/while/add_4_grad/ShapeShapelstm_1/while/mul_1*
T0*%
_class
loc:@lstm_1/while/add_4*
out_type0*
_output_shapes
:
П
7training/Adam/gradients/lstm_1/while/add_4_grad/Shape_1Const^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_4*
valueB *
dtype0*
_output_shapes
: 
Э
Etraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopV27training/Adam/gradients/lstm_1/while/add_4_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/add_4
Н
Ktraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/add_4*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/Const*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/add_4*

stack_name 
Ч
Ktraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc*%
_class
loc:@lstm_1/while/add_4*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0
с
Qtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/add_4_grad/Shape^training/Adam/gradients/Add*
_output_shapes
:*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/add_4
Ј
Ptraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_4*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_4*
parallel_iterations *
is_constant(
Г
3training/Adam/gradients/lstm_1/while/add_4_grad/SumSumItraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/ReshapeEtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs*
T0*%
_class
loc:@lstm_1/while/add_4*
_output_shapes
:*
	keep_dims( *

Tidx0
А
7training/Adam/gradients/lstm_1/while/add_4_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_4_grad/SumPtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/add_4*
Tshape0*'
_output_shapes
:џџџџџџџџџ
З
5training/Adam/gradients/lstm_1/while/add_4_grad/Sum_1SumItraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/ReshapeGtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_4*
_output_shapes
:

9training/Adam/gradients/lstm_1/while/add_4_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_4_grad/Sum_17training/Adam/gradients/lstm_1/while/add_4_grad/Shape_1*
_output_shapes
: *
T0*%
_class
loc:@lstm_1/while/add_4*
Tshape0
Ќ
5training/Adam/gradients/lstm_1/while/add_2_grad/ShapeShapelstm_1/while/mul*
T0*%
_class
loc:@lstm_1/while/add_2*
out_type0*
_output_shapes
:
П
7training/Adam/gradients/lstm_1/while/add_2_grad/Shape_1Const^training/Adam/gradients/Sub*
dtype0*
_output_shapes
: *%
_class
loc:@lstm_1/while/add_2*
valueB 
Э
Etraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopV27training/Adam/gradients/lstm_1/while/add_2_grad/Shape_1*
T0*%
_class
loc:@lstm_1/while/add_2*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/add_2*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/add_2*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_2*
parallel_iterations 
с
Qtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/add_2_grad/Shape^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/add_2*
_output_shapes
:*
swap_memory(
Ј
Ptraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_2*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_2*
parallel_iterations 
Б
3training/Adam/gradients/lstm_1/while/add_2_grad/SumSumGtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/ReshapeEtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/add_2*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
А
7training/Adam/gradients/lstm_1/while/add_2_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_2_grad/SumPtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/add_2*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Е
5training/Adam/gradients/lstm_1/while/add_2_grad/Sum_1SumGtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/ReshapeGtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@lstm_1/while/add_2*
_output_shapes
:*
	keep_dims( *

Tidx0

9training/Adam/gradients/lstm_1/while/add_2_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_2_grad/Sum_17training/Adam/gradients/lstm_1/while/add_2_grad/Shape_1*
T0*%
_class
loc:@lstm_1/while/add_2*
Tshape0*
_output_shapes
: 
ю
?training/Adam/gradients/lstm_1/while/BiasAdd_2_grad/BiasAddGradBiasAddGrad7training/Adam/gradients/lstm_1/while/add_5_grad/Reshape*)
_class
loc:@lstm_1/while/BiasAdd_2*
data_formatNHWC*
_output_shapes
:*
T0
С
9training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMulMatMul9training/Adam/gradients/lstm_1/while/add_5_grad/Reshape_1?training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMul/Enter*
T0*(
_class
loc:@lstm_1/while/MatMul_6*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMul/EnterEnterlstm_1/strided_slice_6*
is_constant(*
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_6*
parallel_iterations 
С
;training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV29training/Adam/gradients/lstm_1/while/add_5_grad/Reshape_1*
T0*(
_class
loc:@lstm_1/while/MatMul_6*
_output_shapes

:*
transpose_a(*
transpose_b( 
П
9training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMulMatMul7training/Adam/gradients/lstm_1/while/add_7_grad/Reshape?training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul/Enter*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b(*
T0*(
_class
loc:@lstm_1/while/MatMul_3
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul/EnterEnterlstm_1/strided_slice_3*
is_constant(*
_output_shapes

:@*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_3*
parallel_iterations 
П
;training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV27training/Adam/gradients/lstm_1/while/add_7_grad/Reshape*
transpose_b( *
T0*(
_class
loc:@lstm_1/while/MatMul_3*
_output_shapes

:@*
transpose_a(
л
Atraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/ConstConst*M
_classC
Aloc:@lstm_1/while/MatMul_3#loc:@lstm_1/while/TensorArrayReadV3*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Atraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_accStackV2Atraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*M
_classC
Aloc:@lstm_1/while/MatMul_3#loc:@lstm_1/while/TensorArrayReadV3
Ж
Atraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/EnterEnterAtraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_3*
parallel_iterations *
is_constant(
Ц
Gtraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPushV2StackPushV2Atraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/Enterlstm_1/while/TensorArrayReadV3^training/Adam/gradients/Add*'
_output_shapes
:џџџџџџџџџ@*
swap_memory(*
T0*(
_class
loc:@lstm_1/while/MatMul_3
Є
Ftraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV2
StackPopV2Ltraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV2/Enter^training/Adam/gradients/Sub*(
_class
loc:@lstm_1/while/MatMul_3*'
_output_shapes
:џџџџџџџџџ@*
	elem_type0
й
Ltraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV2/EnterEnterAtraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_3*
parallel_iterations 
Н
?training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_accConst*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter*
valueB*    *
dtype0*
_output_shapes
:
г
Atraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_1Enter?training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter*
parallel_iterations *
is_constant( *
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
З
Atraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_2MergeAtraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_1Gtraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/NextIteration*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter*
N*
_output_shapes

:: *
T0

@training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/SwitchSwitchAtraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter* 
_output_shapes
::

=training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/AddAddBtraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/Switch:1?training/Adam/gradients/lstm_1/while/BiasAdd_3_grad/BiasAddGrad*
_output_shapes
:*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter
э
Gtraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/NextIterationNextIteration=training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/Add*
_output_shapes
:*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter
с
Atraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_3Exit@training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/Switch*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter*
_output_shapes
:
У
>training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_accConst*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*
valueB*    *
dtype0*
_output_shapes

:
д
@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc*
T0*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*
parallel_iterations *
is_constant( *
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
З
@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/NextIteration*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*
N* 
_output_shapes
:: *
T0

?training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*(
_output_shapes
::

<training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1*
T0*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*
_output_shapes

:
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/Add*
T0*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*
_output_shapes

:
т
@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/Switch*
T0*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*
_output_shapes

:
Н
5training/Adam/gradients/lstm_1/while/mul_1_grad/ShapeConst^training/Adam/gradients/Sub*
_output_shapes
: *%
_class
loc:@lstm_1/while/mul_1*
valueB *
dtype0
А
7training/Adam/gradients/lstm_1/while/mul_1_grad/Shape_1Shapelstm_1/while/add_3*
_output_shapes
:*
T0*%
_class
loc:@lstm_1/while/mul_1*
out_type0
Ы
Etraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/lstm_1/while/mul_1_grad/ShapePtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopV2*%
_class
loc:@lstm_1/while/mul_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Н
Ktraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/mul_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/mul_1*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_1*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
у
Qtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/Enter7training/Adam/gradients/lstm_1/while/mul_1_grad/Shape_1^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_1*
_output_shapes
:*
swap_memory(
Ј
Ptraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
	elem_type0*%
_class
loc:@lstm_1/while/mul_1*
_output_shapes
:
ъ
Vtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_1*
parallel_iterations 

3training/Adam/gradients/lstm_1/while/mul_1_grad/MulMul7training/Adam/gradients/lstm_1/while/add_4_grad/Reshape>training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_1*'
_output_shapes
:џџџџџџџџџ
Ф
9training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/ConstConst*>
_class4
2loc:@lstm_1/while/add_3loc:@lstm_1/while/mul_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
ў
9training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/f_accStackV29training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/Const*>
_class4
2loc:@lstm_1/while/add_3loc:@lstm_1/while/mul_1*

stack_name *
_output_shapes
:*
	elem_type0
Ѓ
9training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/EnterEnter9training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_1*
parallel_iterations 
Ї
?training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPushV2StackPushV29training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/Enterlstm_1/while/add_3^training/Adam/gradients/Add*%
_class
loc:@lstm_1/while/mul_1*'
_output_shapes
:џџџџџџџџџ*
swap_memory(*
T0

>training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPopV2
StackPopV2Dtraining/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*'
_output_shapes
:џџџџџџџџџ*
	elem_type0*%
_class
loc:@lstm_1/while/mul_1
Ц
Dtraining/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPopV2/EnterEnter9training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_1*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context

3training/Adam/gradients/lstm_1/while/mul_1_grad/SumSum3training/Adam/gradients/lstm_1/while/mul_1_grad/MulEtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/mul_1*
_output_shapes
:

7training/Adam/gradients/lstm_1/while/mul_1_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/mul_1_grad/Sum5training/Adam/gradients/lstm_1/while/mul_1_grad/Shape*
T0*%
_class
loc:@lstm_1/while/mul_1*
Tshape0*
_output_shapes
: 

5training/Adam/gradients/lstm_1/while/mul_1_grad/Mul_1Mul;training/Adam/gradients/lstm_1/while/mul_1_grad/Mul_1/Const7training/Adam/gradients/lstm_1/while/add_4_grad/Reshape*
T0*%
_class
loc:@lstm_1/while/mul_1*'
_output_shapes
:џџџџџџџџџ
Х
;training/Adam/gradients/lstm_1/while/mul_1_grad/Mul_1/ConstConst^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_1*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
Ѓ
5training/Adam/gradients/lstm_1/while/mul_1_grad/Sum_1Sum5training/Adam/gradients/lstm_1/while/mul_1_grad/Mul_1Gtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/mul_1*
_output_shapes
:
Д
9training/Adam/gradients/lstm_1/while/mul_1_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/mul_1_grad/Sum_1Ptraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/mul_1*
Tshape0
Й
3training/Adam/gradients/lstm_1/while/mul_grad/ShapeConst^training/Adam/gradients/Sub*#
_class
loc:@lstm_1/while/mul*
valueB *
dtype0*
_output_shapes
: 
Ќ
5training/Adam/gradients/lstm_1/while/mul_grad/Shape_1Shapelstm_1/while/add_1*
T0*#
_class
loc:@lstm_1/while/mul*
out_type0*
_output_shapes
:
У
Ctraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgsBroadcastGradientArgs3training/Adam/gradients/lstm_1/while/mul_grad/ShapeNtraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*#
_class
loc:@lstm_1/while/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Й
Itraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/ConstConst*#
_class
loc:@lstm_1/while/mul*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Itraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_accStackV2Itraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/Const*#
_class
loc:@lstm_1/while/mul*

stack_name *
_output_shapes
:*
	elem_type0
С
Itraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/EnterEnterItraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_acc*
T0*#
_class
loc:@lstm_1/while/mul*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
л
Otraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Itraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/mul_grad/Shape_1^training/Adam/gradients/Add*
_output_shapes
:*
swap_memory(*
T0*#
_class
loc:@lstm_1/while/mul
Ђ
Ntraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ttraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*#
_class
loc:@lstm_1/while/mul
ф
Ttraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterItraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*#
_class
loc:@lstm_1/while/mul*
parallel_iterations *
is_constant(

1training/Adam/gradients/lstm_1/while/mul_grad/MulMul7training/Adam/gradients/lstm_1/while/add_2_grad/Reshape<training/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPopV2*
T0*#
_class
loc:@lstm_1/while/mul*'
_output_shapes
:џџџџџџџџџ
Р
7training/Adam/gradients/lstm_1/while/mul_grad/Mul/ConstConst*<
_class2
0loc:@lstm_1/while/add_1loc:@lstm_1/while/mul*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
ј
7training/Adam/gradients/lstm_1/while/mul_grad/Mul/f_accStackV27training/Adam/gradients/lstm_1/while/mul_grad/Mul/Const*<
_class2
0loc:@lstm_1/while/add_1loc:@lstm_1/while/mul*

stack_name *
_output_shapes
:*
	elem_type0

7training/Adam/gradients/lstm_1/while/mul_grad/Mul/EnterEnter7training/Adam/gradients/lstm_1/while/mul_grad/Mul/f_acc*
T0*#
_class
loc:@lstm_1/while/mul*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
Ё
=training/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPushV2StackPushV27training/Adam/gradients/lstm_1/while/mul_grad/Mul/Enterlstm_1/while/add_1^training/Adam/gradients/Add*
T0*#
_class
loc:@lstm_1/while/mul*'
_output_shapes
:џџџџџџџџџ*
swap_memory(

<training/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPopV2
StackPopV2Btraining/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*#
_class
loc:@lstm_1/while/mul*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
Р
Btraining/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPopV2/EnterEnter7training/Adam/gradients/lstm_1/while/mul_grad/Mul/f_acc*
T0*#
_class
loc:@lstm_1/while/mul*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context

1training/Adam/gradients/lstm_1/while/mul_grad/SumSum1training/Adam/gradients/lstm_1/while/mul_grad/MulCtraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs*#
_class
loc:@lstm_1/while/mul*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ќ
5training/Adam/gradients/lstm_1/while/mul_grad/ReshapeReshape1training/Adam/gradients/lstm_1/while/mul_grad/Sum3training/Adam/gradients/lstm_1/while/mul_grad/Shape*
T0*#
_class
loc:@lstm_1/while/mul*
Tshape0*
_output_shapes
: 

3training/Adam/gradients/lstm_1/while/mul_grad/Mul_1Mul9training/Adam/gradients/lstm_1/while/mul_grad/Mul_1/Const7training/Adam/gradients/lstm_1/while/add_2_grad/Reshape*
T0*#
_class
loc:@lstm_1/while/mul*'
_output_shapes
:џџџџџџџџџ
С
9training/Adam/gradients/lstm_1/while/mul_grad/Mul_1/ConstConst^training/Adam/gradients/Sub*#
_class
loc:@lstm_1/while/mul*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 

3training/Adam/gradients/lstm_1/while/mul_grad/Sum_1Sum3training/Adam/gradients/lstm_1/while/mul_grad/Mul_1Etraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*#
_class
loc:@lstm_1/while/mul
Ќ
7training/Adam/gradients/lstm_1/while/mul_grad/Reshape_1Reshape3training/Adam/gradients/lstm_1/while/mul_grad/Sum_1Ntraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*#
_class
loc:@lstm_1/while/mul*
Tshape0*'
_output_shapes
:џџџџџџџџџ
П
9training/Adam/gradients/lstm_1/while/MatMul_2_grad/MatMulMatMul7training/Adam/gradients/lstm_1/while/add_5_grad/Reshape?training/Adam/gradients/lstm_1/while/MatMul_2_grad/MatMul/Enter*
T0*(
_class
loc:@lstm_1/while/MatMul_2*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b(
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_2_grad/MatMul/EnterEnterlstm_1/strided_slice_2*
_output_shapes

:@*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_2*
parallel_iterations *
is_constant(
П
;training/Adam/gradients/lstm_1/while/MatMul_2_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV27training/Adam/gradients/lstm_1/while/add_5_grad/Reshape*
T0*(
_class
loc:@lstm_1/while/MatMul_2*
_output_shapes

:@*
transpose_a(*
transpose_b( 
Н
?training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_accConst*
dtype0*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter*
valueB*    
г
Atraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc_1Enter?training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc*
is_constant( *
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter*
parallel_iterations 
З
Atraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc_2MergeAtraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc_1Gtraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/NextIteration*
N*
_output_shapes

:: *
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter

@training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/SwitchSwitchAtraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2* 
_output_shapes
::*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter

=training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/AddAddBtraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/Switch:1?training/Adam/gradients/lstm_1/while/BiasAdd_2_grad/BiasAddGrad*
_output_shapes
:*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter
э
Gtraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/NextIterationNextIteration=training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/Add*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter*
_output_shapes
:
с
Atraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc_3Exit@training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/Switch*
_output_shapes
:*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter
У
>training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_accConst*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*
valueB*    *
dtype0*
_output_shapes

:
д
@training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc*
is_constant( *
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*
parallel_iterations 
З
@training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/NextIteration* 
_output_shapes
:: *
T0*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*
N

?training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*(
_output_shapes
::

<training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMul_1*
T0*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*
_output_shapes

:
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/Add*
T0*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*
_output_shapes

:
т
@training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/Switch*
T0*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*
_output_shapes

:
У
>training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_accConst*
dtype0*
_output_shapes

:@*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
valueB@*    
д
@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc*
is_constant( *
_output_shapes

:@*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
parallel_iterations 
З
@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/NextIteration*
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
N* 
_output_shapes
:@: 

?training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*(
_output_shapes
:@:@

<training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1*
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
_output_shapes

:@
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/Add*
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
_output_shapes

:@
т
@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/Switch*
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
_output_shapes

:@
А
:training/Adam/gradients/lstm_1/strided_slice_11_grad/ShapeConst**
_class 
loc:@lstm_1/strided_slice_11*
valueB:*
dtype0*
_output_shapes
:
ћ
Etraining/Adam/gradients/lstm_1/strided_slice_11_grad/StridedSliceGradStridedSliceGrad:training/Adam/gradients/lstm_1/strided_slice_11_grad/Shapelstm_1/strided_slice_11/stacklstm_1/strided_slice_11/stack_1lstm_1/strided_slice_11/stack_2Atraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_3*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
Index0*
T0**
_class 
loc:@lstm_1/strided_slice_11
Е
9training/Adam/gradients/lstm_1/strided_slice_7_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_7*
valueB"      *
dtype0*
_output_shapes
:
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_7_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_7_grad/Shapelstm_1/strided_slice_7/stacklstm_1/strided_slice_7/stack_1lstm_1/strided_slice_7/stack_2@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_3*
_output_shapes

:*
T0*
Index0*)
_class
loc:@lstm_1/strided_slice_7*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
В
5training/Adam/gradients/lstm_1/while/add_3_grad/ShapeShapelstm_1/while/BiasAdd_1*%
_class
loc:@lstm_1/while/add_3*
out_type0*
_output_shapes
:*
T0
Г
7training/Adam/gradients/lstm_1/while/add_3_grad/Shape_1Shapelstm_1/while/MatMul_5*
T0*%
_class
loc:@lstm_1/while/add_3*
out_type0*
_output_shapes
:
ш
Etraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2Rtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/add_3*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/add_3*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/add_3*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_3*
parallel_iterations 
с
Qtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/add_3_grad/Shape^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/add_3*
_output_shapes
:*
swap_memory(
Ј
Ptraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_3*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc*%
_class
loc:@lstm_1/while/add_3*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0
П
Mtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/Const_1Const*%
_class
loc:@lstm_1/while/add_3*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Mtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/Const_1*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/add_3*

stack_name 
Ы
Mtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc_1*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_3*
parallel_iterations 
ч
Straining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/add_3_grad/Shape_1^training/Adam/gradients/Add*
_output_shapes
:*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/add_3
Ќ
Rtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*
	elem_type0*%
_class
loc:@lstm_1/while/add_3*
_output_shapes
:
ю
Xtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterMtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc_1*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_3*
parallel_iterations 
Ѓ
3training/Adam/gradients/lstm_1/while/add_3_grad/SumSum9training/Adam/gradients/lstm_1/while/mul_1_grad/Reshape_1Etraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_3*
_output_shapes
:
А
7training/Adam/gradients/lstm_1/while/add_3_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_3_grad/SumPtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2*%
_class
loc:@lstm_1/while/add_3*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
Ї
5training/Adam/gradients/lstm_1/while/add_3_grad/Sum_1Sum9training/Adam/gradients/lstm_1/while/mul_1_grad/Reshape_1Gtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_3
Ж
9training/Adam/gradients/lstm_1/while/add_3_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_3_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/add_3*
Tshape0*'
_output_shapes
:џџџџџџџџџ
А
5training/Adam/gradients/lstm_1/while/add_1_grad/ShapeShapelstm_1/while/BiasAdd*
T0*%
_class
loc:@lstm_1/while/add_1*
out_type0*
_output_shapes
:
Г
7training/Adam/gradients/lstm_1/while/add_1_grad/Shape_1Shapelstm_1/while/MatMul_4*
T0*%
_class
loc:@lstm_1/while/add_1*
out_type0*
_output_shapes
:
ш
Etraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2Rtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/add_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/ConstConst*%
_class
loc:@lstm_1/while/add_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/add_1*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc*
T0*%
_class
loc:@lstm_1/while/add_1*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
с
Qtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/add_1_grad/Shape^training/Adam/gradients/Add*
_output_shapes
:*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/add_1
Ј
Ptraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_1*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_1*
parallel_iterations 
П
Mtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Const_1Const*%
_class
loc:@lstm_1/while/add_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Mtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Const_1*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/add_1*

stack_name 
Ы
Mtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc_1*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_1*
parallel_iterations 
ч
Straining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/add_1_grad/Shape_1^training/Adam/gradients/Add*
_output_shapes
:*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/add_1
Ќ
Rtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_1*
_output_shapes
:*
	elem_type0
ю
Xtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterMtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*%
_class
loc:@lstm_1/while/add_1*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
Ё
3training/Adam/gradients/lstm_1/while/add_1_grad/SumSum7training/Adam/gradients/lstm_1/while/mul_grad/Reshape_1Etraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs*
T0*%
_class
loc:@lstm_1/while/add_1*
_output_shapes
:*
	keep_dims( *

Tidx0
А
7training/Adam/gradients/lstm_1/while/add_1_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_1_grad/SumPtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/add_1*
Tshape0
Ѕ
5training/Adam/gradients/lstm_1/while/add_1_grad/Sum_1Sum7training/Adam/gradients/lstm_1/while/mul_grad/Reshape_1Gtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_1*
_output_shapes
:
Ж
9training/Adam/gradients/lstm_1/while/add_1_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_1_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/add_1*
Tshape0*'
_output_shapes
:џџџџџџџџџ
У
>training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_accConst*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
valueB@*    *
dtype0*
_output_shapes

:@
д
@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
parallel_iterations *
is_constant( *
_output_shapes

:@*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0
З
@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/NextIteration*
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
N* 
_output_shapes
:@: 

?training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*(
_output_shapes
:@:@

<training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_2_grad/MatMul_1*
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
_output_shapes

:@
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/Add*
_output_shapes

:@*
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter
т
@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
_output_shapes

:@*
T0
А
:training/Adam/gradients/lstm_1/strided_slice_10_grad/ShapeConst**
_class 
loc:@lstm_1/strided_slice_10*
valueB:*
dtype0*
_output_shapes
:
ћ
Etraining/Adam/gradients/lstm_1/strided_slice_10_grad/StridedSliceGradStridedSliceGrad:training/Adam/gradients/lstm_1/strided_slice_10_grad/Shapelstm_1/strided_slice_10/stacklstm_1/strided_slice_10/stack_1lstm_1/strided_slice_10/stack_2Atraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc_3*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0**
_class 
loc:@lstm_1/strided_slice_10
Е
9training/Adam/gradients/lstm_1/strided_slice_6_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_6*
valueB"      *
dtype0*
_output_shapes
:
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_6_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_6_grad/Shapelstm_1/strided_slice_6/stacklstm_1/strided_slice_6/stack_1lstm_1/strided_slice_6/stack_2@training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc_3*)
_class
loc:@lstm_1/strided_slice_6*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:*
T0*
Index0
Е
9training/Adam/gradients/lstm_1/strided_slice_3_grad/ShapeConst*
dtype0*
_output_shapes
:*)
_class
loc:@lstm_1/strided_slice_3*
valueB"@      
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_3_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_3_grad/Shapelstm_1/strided_slice_3/stacklstm_1/strided_slice_3/stack_1lstm_1/strided_slice_3/stack_2@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_3*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:@*
T0*
Index0*)
_class
loc:@lstm_1/strided_slice_3
ю
?training/Adam/gradients/lstm_1/while/BiasAdd_1_grad/BiasAddGradBiasAddGrad7training/Adam/gradients/lstm_1/while/add_3_grad/Reshape*
T0*)
_class
loc:@lstm_1/while/BiasAdd_1*
data_formatNHWC*
_output_shapes
:
С
9training/Adam/gradients/lstm_1/while/MatMul_5_grad/MatMulMatMul9training/Adam/gradients/lstm_1/while/add_3_grad/Reshape_1?training/Adam/gradients/lstm_1/while/MatMul_5_grad/MatMul/Enter*
T0*(
_class
loc:@lstm_1/while/MatMul_5*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_5_grad/MatMul/EnterEnterlstm_1/strided_slice_5*
is_constant(*
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_5*
parallel_iterations 
С
;training/Adam/gradients/lstm_1/while/MatMul_5_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV29training/Adam/gradients/lstm_1/while/add_3_grad/Reshape_1*
T0*(
_class
loc:@lstm_1/while/MatMul_5*
_output_shapes

:*
transpose_a(*
transpose_b( 
ъ
=training/Adam/gradients/lstm_1/while/BiasAdd_grad/BiasAddGradBiasAddGrad7training/Adam/gradients/lstm_1/while/add_1_grad/Reshape*
T0*'
_class
loc:@lstm_1/while/BiasAdd*
data_formatNHWC*
_output_shapes
:
С
9training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMulMatMul9training/Adam/gradients/lstm_1/while/add_1_grad/Reshape_1?training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMul/Enter*
T0*(
_class
loc:@lstm_1/while/MatMul_4*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMul/EnterEnterlstm_1/strided_slice_4*
parallel_iterations *
is_constant(*
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_4
С
;training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV29training/Adam/gradients/lstm_1/while/add_1_grad/Reshape_1*
transpose_b( *
T0*(
_class
loc:@lstm_1/while/MatMul_4*
_output_shapes

:*
transpose_a(
Е
9training/Adam/gradients/lstm_1/strided_slice_2_grad/ShapeConst*
_output_shapes
:*)
_class
loc:@lstm_1/strided_slice_2*
valueB"@      *
dtype0
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_2_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_2_grad/Shapelstm_1/strided_slice_2/stacklstm_1/strided_slice_2/stack_1lstm_1/strided_slice_2/stack_2@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_3*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:@*
Index0*
T0*)
_class
loc:@lstm_1/strided_slice_2*
shrink_axis_mask 
П
9training/Adam/gradients/lstm_1/while/MatMul_1_grad/MatMulMatMul7training/Adam/gradients/lstm_1/while/add_3_grad/Reshape?training/Adam/gradients/lstm_1/while/MatMul_1_grad/MatMul/Enter*
T0*(
_class
loc:@lstm_1/while/MatMul_1*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b(
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_1_grad/MatMul/EnterEnterlstm_1/strided_slice_1*
is_constant(*
_output_shapes

:@*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_1*
parallel_iterations 
П
;training/Adam/gradients/lstm_1/while/MatMul_1_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV27training/Adam/gradients/lstm_1/while/add_3_grad/Reshape*
T0*(
_class
loc:@lstm_1/while/MatMul_1*
_output_shapes

:@*
transpose_a(*
transpose_b( 
Н
?training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_accConst*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter*
valueB*    *
dtype0
г
Atraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_1Enter?training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter*
parallel_iterations *
is_constant( *
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
З
Atraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_2MergeAtraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_1Gtraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/NextIteration*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter*
N*
_output_shapes

:: 

@training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/SwitchSwitchAtraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2* 
_output_shapes
::*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter

=training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/AddAddBtraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/Switch:1?training/Adam/gradients/lstm_1/while/BiasAdd_1_grad/BiasAddGrad*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter*
_output_shapes
:*
T0
э
Gtraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/NextIterationNextIteration=training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/Add*
_output_shapes
:*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter
с
Atraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_3Exit@training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/Switch*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter*
_output_shapes
:
У
>training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_accConst*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter*
valueB*    *
dtype0*
_output_shapes

:
д
@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc*
T0*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter*
parallel_iterations *
is_constant( *
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
З
@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/NextIteration*
T0*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter*
N* 
_output_shapes
:: 

?training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter*(
_output_shapes
::

<training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_5_grad/MatMul_1*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter*
_output_shapes

:*
T0
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/Add*
T0*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter*
_output_shapes

:
т
@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/Switch*
_output_shapes

:*
T0*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter
Й
7training/Adam/gradients/lstm_1/while/MatMul_grad/MatMulMatMul7training/Adam/gradients/lstm_1/while/add_1_grad/Reshape=training/Adam/gradients/lstm_1/while/MatMul_grad/MatMul/Enter*
T0*&
_class
loc:@lstm_1/while/MatMul*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b(

=training/Adam/gradients/lstm_1/while/MatMul_grad/MatMul/EnterEnterlstm_1/strided_slice*
is_constant(*
_output_shapes

:@*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*&
_class
loc:@lstm_1/while/MatMul*
parallel_iterations 
Л
9training/Adam/gradients/lstm_1/while/MatMul_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV27training/Adam/gradients/lstm_1/while/add_1_grad/Reshape*
transpose_b( *
T0*&
_class
loc:@lstm_1/while/MatMul*
_output_shapes

:@*
transpose_a(
Й
=training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_accConst*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter*
valueB*    *
dtype0*
_output_shapes
:
Э
?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_1Enter=training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc*
is_constant( *
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter*
parallel_iterations 
Џ
?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_2Merge?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_1Etraining/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/NextIteration*
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter*
N*
_output_shapes

:: 

>training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/SwitchSwitch?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter* 
_output_shapes
::

;training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/AddAdd@training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/Switch:1=training/Adam/gradients/lstm_1/while/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter
ч
Etraining/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/NextIterationNextIteration;training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/Add*
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter*
_output_shapes
:
л
?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_3Exit>training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/Switch*
_output_shapes
:*
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter
ї
training/Adam/gradients/AddN_7AddN9training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul9training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMul9training/Adam/gradients/lstm_1/while/MatMul_5_grad/MatMul9training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMul*
N*'
_output_shapes
:џџџџџџџџџ*
T0*(
_class
loc:@lstm_1/while/MatMul_7
У
>training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_accConst*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter*
valueB*    *
dtype0*
_output_shapes

:
д
@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc*
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter*
parallel_iterations *
is_constant( 
З
@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/NextIteration*
T0*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter*
N* 
_output_shapes
:: 

?training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*(
_output_shapes
::*
T0*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter

<training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMul_1*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter*
_output_shapes

:*
T0
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/Add*
T0*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter*
_output_shapes

:
т
@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/Switch*
_output_shapes

:*
T0*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter
У
>training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_accConst*
dtype0*
_output_shapes

:@*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
valueB@*    
д
@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc*
_output_shapes

:@*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
parallel_iterations *
is_constant( 
З
@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/NextIteration*
N* 
_output_shapes
:@: *
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter

?training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*(
_output_shapes
:@:@

<training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_1_grad/MatMul_1*
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
_output_shapes

:@
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/Add*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
_output_shapes

:@*
T0
т
@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/Switch*
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
_output_shapes

:@
Ў
9training/Adam/gradients/lstm_1/strided_slice_9_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_9*
valueB:*
dtype0*
_output_shapes
:
ѕ
Dtraining/Adam/gradients/lstm_1/strided_slice_9_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_9_grad/Shapelstm_1/strided_slice_9/stacklstm_1/strided_slice_9/stack_1lstm_1/strided_slice_9/stack_2Atraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_3*
T0*
Index0*)
_class
loc:@lstm_1/strided_slice_9*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:
Е
9training/Adam/gradients/lstm_1/strided_slice_5_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_5*
valueB"      *
dtype0*
_output_shapes
:
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_5_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_5_grad/Shapelstm_1/strided_slice_5/stacklstm_1/strided_slice_5/stack_1lstm_1/strided_slice_5/stack_2@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_3*)
_class
loc:@lstm_1/strided_slice_5*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:*
T0*
Index0
ѕ
training/Adam/gradients/AddN_8AddN9training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul9training/Adam/gradients/lstm_1/while/MatMul_2_grad/MatMul9training/Adam/gradients/lstm_1/while/MatMul_1_grad/MatMul7training/Adam/gradients/lstm_1/while/MatMul_grad/MatMul*
T0*(
_class
loc:@lstm_1/while/MatMul_3*
N*'
_output_shapes
:џџџџџџџџџ@
§
]training/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3ctraining/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enteretraining/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^training/Adam/gradients/Sub*\
_classR
P#loc:@lstm_1/while/TensorArrayReadV3)loc:@lstm_1/while/TensorArrayReadV3/Enter*#
sourcetraining/Adam/gradients*
_output_shapes

:: 
ї
ctraining/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlstm_1/TensorArray_1*
T0*\
_classR
P#loc:@lstm_1/while/TensorArrayReadV3)loc:@lstm_1/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
Ђ
etraining/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterAlstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*\
_classR
P#loc:@lstm_1/while/TensorArrayReadV3)loc:@lstm_1/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(*
_output_shapes
: *B

frame_name42training/Adam/gradients/lstm_1/while/while_context
Ћ
Ytraining/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityetraining/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^^training/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*\
_classR
P#loc:@lstm_1/while/TensorArrayReadV3)loc:@lstm_1/while/TensorArrayReadV3/Enter

_training/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3]training/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3jtraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2training/Adam/gradients/AddN_8Ytraining/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0*1
_class'
%#loc:@lstm_1/while/TensorArrayReadV3
П
<training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_accConst*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
valueB@*    *
dtype0*
_output_shapes

:@
Ю
>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_1Enter<training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc*
is_constant( *
_output_shapes

:@*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
parallel_iterations 
Џ
>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_2Merge>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_1Dtraining/Adam/gradients/lstm_1/while/MatMul/Enter_grad/NextIteration*
N* 
_output_shapes
:@: *
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter

=training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/SwitchSwitch>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*(
_output_shapes
:@:@

:training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/AddAdd?training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/Switch:19training/Adam/gradients/lstm_1/while/MatMul_grad/MatMul_1*
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
_output_shapes

:@
ш
Dtraining/Adam/gradients/lstm_1/while/MatMul/Enter_grad/NextIterationNextIteration:training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/Add*
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
_output_shapes

:@
м
>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_3Exit=training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/Switch*
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
_output_shapes

:@
Ў
9training/Adam/gradients/lstm_1/strided_slice_8_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_8*
valueB:*
dtype0*
_output_shapes
:
ѓ
Dtraining/Adam/gradients/lstm_1/strided_slice_8_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_8_grad/Shapelstm_1/strided_slice_8/stacklstm_1/strided_slice_8/stack_1lstm_1/strided_slice_8/stack_2?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_3*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0*)
_class
loc:@lstm_1/strided_slice_8
Ю
Btraining/Adam/gradients/lstm_1/while/Switch_3_grad_1/NextIterationNextIterationtraining/Adam/gradients/AddN_7*
T0*'
_class
loc:@lstm_1/while/Merge_3*'
_output_shapes
:џџџџџџџџџ
Е
9training/Adam/gradients/lstm_1/strided_slice_4_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_4*
valueB"      *
dtype0*
_output_shapes
:
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_4_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_4_grad/Shapelstm_1/strided_slice_4/stacklstm_1/strided_slice_4/stack_1lstm_1/strided_slice_4/stack_2@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_3*
T0*
Index0*)
_class
loc:@lstm_1/strided_slice_4*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:
Е
9training/Adam/gradients/lstm_1/strided_slice_1_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_1*
valueB"@      *
dtype0*
_output_shapes
:
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_1_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_1_grad/Shapelstm_1/strided_slice_1/stacklstm_1/strided_slice_1/stack_1lstm_1/strided_slice_1/stack_2@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_3*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:@*
Index0*
T0*)
_class
loc:@lstm_1/strided_slice_1*
shrink_axis_mask 
Щ
Itraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
dtype0*
_output_shapes
: *9
_class/
-+loc:@lstm_1/while/TensorArrayReadV3/Enter_1*
valueB
 *    
э
Ktraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterItraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*9
_class/
-+loc:@lstm_1/while/TensorArrayReadV3/Enter_1*
parallel_iterations *
is_constant( *
_output_shapes
: *B

frame_name42training/Adam/gradients/lstm_1/while/while_context
л
Ktraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeKtraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Qtraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*9
_class/
-+loc:@lstm_1/while/TensorArrayReadV3/Enter_1*
N*
_output_shapes
: : 
Ђ
Jtraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchKtraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_2!training/Adam/gradients/b_count_2*
_output_shapes
: : *
T0*9
_class/
-+loc:@lstm_1/while/TensorArrayReadV3/Enter_1
й
Gtraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/AddAddLtraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/Switch:1_training/Adam/gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0*9
_class/
-+loc:@lstm_1/while/TensorArrayReadV3/Enter_1

Qtraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIterationGtraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*9
_class/
-+loc:@lstm_1/while/TensorArrayReadV3/Enter_1*
_output_shapes
: 
ћ
Ktraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitJtraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0*9
_class/
-+loc:@lstm_1/while/TensorArrayReadV3/Enter_1*
_output_shapes
: 
Б
7training/Adam/gradients/lstm_1/strided_slice_grad/ShapeConst*'
_class
loc:@lstm_1/strided_slice*
valueB"@      *
dtype0*
_output_shapes
:
ъ
Btraining/Adam/gradients/lstm_1/strided_slice_grad/StridedSliceGradStridedSliceGrad7training/Adam/gradients/lstm_1/strided_slice_grad/Shapelstm_1/strided_slice/stacklstm_1/strided_slice/stack_1lstm_1/strided_slice/stack_2>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_3*
new_axis_mask *
end_mask*
_output_shapes

:@*
Index0*
T0*'
_class
loc:@lstm_1/strided_slice*
shrink_axis_mask *
ellipsis_mask *

begin_mask

training/Adam/gradients/AddN_9AddNEtraining/Adam/gradients/lstm_1/strided_slice_11_grad/StridedSliceGradEtraining/Adam/gradients/lstm_1/strided_slice_10_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_9_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_8_grad/StridedSliceGrad*
T0**
_class 
loc:@lstm_1/strided_slice_11*
N*
_output_shapes
:

training/Adam/gradients/AddN_10AddNDtraining/Adam/gradients/lstm_1/strided_slice_7_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_6_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_5_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_4_grad/StridedSliceGrad*
T0*)
_class
loc:@lstm_1/strided_slice_7*
N*
_output_shapes

:
ќ
training/Adam/gradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lstm_1/TensorArray_1Ktraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*>
_class4
2loc:@lstm_1/TensorArray_1loc:@lstm_1/transpose*#
sourcetraining/Adam/gradients*
_output_shapes

:: 
К
|training/Adam/gradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityKtraining/Adam/gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3^training/Adam/gradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*>
_class4
2loc:@lstm_1/TensorArray_1loc:@lstm_1/transpose
Ђ
rtraining/Adam/gradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3training/Adam/gradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3lstm_1/TensorArrayUnstack/range|training/Adam/gradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*#
_class
loc:@lstm_1/transpose*
dtype0*,
_output_shapes
:Ѕџџџџџџџџџ@*
element_shape:

training/Adam/gradients/AddN_11AddNDtraining/Adam/gradients/lstm_1/strided_slice_3_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_2_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_1_grad/StridedSliceGradBtraining/Adam/gradients/lstm_1/strided_slice_grad/StridedSliceGrad*
_output_shapes

:@*
T0*)
_class
loc:@lstm_1/strided_slice_3*
N
Е
?training/Adam/gradients/lstm_1/transpose_grad/InvertPermutationInvertPermutationlstm_1/transpose/perm*
T0*#
_class
loc:@lstm_1/transpose*
_output_shapes
:
т
7training/Adam/gradients/lstm_1/transpose_grad/transpose	Transposertraining/Adam/gradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3?training/Adam/gradients/lstm_1/transpose_grad/InvertPermutation*
T0*#
_class
loc:@lstm_1/transpose*,
_output_shapes
:џџџџџџџџџЅ@*
Tperm0
р
3training/Adam/gradients/conv1d_2/Relu_grad/ReluGradReluGrad7training/Adam/gradients/lstm_1/transpose_grad/transposeconv1d_2/Relu*
T0* 
_class
loc:@conv1d_2/Relu*,
_output_shapes
:џџџџџџџџџЅ@
Ќ
/training/Adam/gradients/conv1d_2/add_grad/ShapeShapeconv1d_2/convolution/Squeeze*
T0*
_class
loc:@conv1d_2/add*
out_type0*
_output_shapes
:
Ї
1training/Adam/gradients/conv1d_2/add_grad/Shape_1Const*
_class
loc:@conv1d_2/add*!
valueB"      @   *
dtype0*
_output_shapes
:

?training/Adam/gradients/conv1d_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs/training/Adam/gradients/conv1d_2/add_grad/Shape1training/Adam/gradients/conv1d_2/add_grad/Shape_1*
T0*
_class
loc:@conv1d_2/add*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

-training/Adam/gradients/conv1d_2/add_grad/SumSum3training/Adam/gradients/conv1d_2/Relu_grad/ReluGrad?training/Adam/gradients/conv1d_2/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_class
loc:@conv1d_2/add*
_output_shapes
:

1training/Adam/gradients/conv1d_2/add_grad/ReshapeReshape-training/Adam/gradients/conv1d_2/add_grad/Sum/training/Adam/gradients/conv1d_2/add_grad/Shape*,
_output_shapes
:џџџџџџџџџЅ@*
T0*
_class
loc:@conv1d_2/add*
Tshape0

/training/Adam/gradients/conv1d_2/add_grad/Sum_1Sum3training/Adam/gradients/conv1d_2/Relu_grad/ReluGradAtraining/Adam/gradients/conv1d_2/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_class
loc:@conv1d_2/add*
_output_shapes
:
ў
3training/Adam/gradients/conv1d_2/add_grad/Reshape_1Reshape/training/Adam/gradients/conv1d_2/add_grad/Sum_11training/Adam/gradients/conv1d_2/add_grad/Shape_1*
T0*
_class
loc:@conv1d_2/add*
Tshape0*"
_output_shapes
:@
Ы
?training/Adam/gradients/conv1d_2/convolution/Squeeze_grad/ShapeShapeconv1d_2/convolution/Conv2D*
T0*/
_class%
#!loc:@conv1d_2/convolution/Squeeze*
out_type0*
_output_shapes
:
К
Atraining/Adam/gradients/conv1d_2/convolution/Squeeze_grad/ReshapeReshape1training/Adam/gradients/conv1d_2/add_grad/Reshape?training/Adam/gradients/conv1d_2/convolution/Squeeze_grad/Shape*
T0*/
_class%
#!loc:@conv1d_2/convolution/Squeeze*
Tshape0*0
_output_shapes
:џџџџџџџџџЅ@
Ђ
3training/Adam/gradients/conv1d_2/Reshape_grad/ShapeConst*
_output_shapes
:*#
_class
loc:@conv1d_2/Reshape*
valueB:@*
dtype0

5training/Adam/gradients/conv1d_2/Reshape_grad/ReshapeReshape3training/Adam/gradients/conv1d_2/add_grad/Reshape_13training/Adam/gradients/conv1d_2/Reshape_grad/Shape*
_output_shapes
:@*
T0*#
_class
loc:@conv1d_2/Reshape*
Tshape0

?training/Adam/gradients/conv1d_2/convolution/Conv2D_grad/ShapeNShapeNconv1d_2/convolution/ExpandDims!conv1d_2/convolution/ExpandDims_1* 
_output_shapes
::*
T0*.
_class$
" loc:@conv1d_2/convolution/Conv2D*
out_type0*
N
ф
Ltraining/Adam/gradients/conv1d_2/convolution/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput?training/Adam/gradients/conv1d_2/convolution/Conv2D_grad/ShapeN!conv1d_2/convolution/ExpandDims_1Atraining/Adam/gradients/conv1d_2/convolution/Squeeze_grad/Reshape*
	dilations
*
T0*.
_class$
" loc:@conv1d_2/convolution/Conv2D*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*0
_output_shapes
:џџџџџџџџџЌ
м
Mtraining/Adam/gradients/conv1d_2/convolution/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterconv1d_2/convolution/ExpandDimsAtraining/Adam/gradients/conv1d_2/convolution/Conv2D_grad/ShapeN:1Atraining/Adam/gradients/conv1d_2/convolution/Squeeze_grad/Reshape*&
_output_shapes
:@*
	dilations
*
T0*.
_class$
" loc:@conv1d_2/convolution/Conv2D*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
У
Btraining/Adam/gradients/conv1d_2/convolution/ExpandDims_grad/ShapeShapeconv1d_1/Relu*
_output_shapes
:*
T0*2
_class(
&$loc:@conv1d_2/convolution/ExpandDims*
out_type0
к
Dtraining/Adam/gradients/conv1d_2/convolution/ExpandDims_grad/ReshapeReshapeLtraining/Adam/gradients/conv1d_2/convolution/Conv2D_grad/Conv2DBackpropInputBtraining/Adam/gradients/conv1d_2/convolution/ExpandDims_grad/Shape*
T0*2
_class(
&$loc:@conv1d_2/convolution/ExpandDims*
Tshape0*,
_output_shapes
:џџџџџџџџџЌ
Я
Dtraining/Adam/gradients/conv1d_2/convolution/ExpandDims_1_grad/ShapeConst*4
_class*
(&loc:@conv1d_2/convolution/ExpandDims_1*!
valueB"      @   *
dtype0*
_output_shapes
:
з
Ftraining/Adam/gradients/conv1d_2/convolution/ExpandDims_1_grad/ReshapeReshapeMtraining/Adam/gradients/conv1d_2/convolution/Conv2D_grad/Conv2DBackpropFilterDtraining/Adam/gradients/conv1d_2/convolution/ExpandDims_1_grad/Shape*
T0*4
_class*
(&loc:@conv1d_2/convolution/ExpandDims_1*
Tshape0*"
_output_shapes
:@
э
3training/Adam/gradients/conv1d_1/Relu_grad/ReluGradReluGradDtraining/Adam/gradients/conv1d_2/convolution/ExpandDims_grad/Reshapeconv1d_1/Relu* 
_class
loc:@conv1d_1/Relu*,
_output_shapes
:џџџџџџџџџЌ*
T0
Ќ
/training/Adam/gradients/conv1d_1/add_grad/ShapeShapeconv1d_1/convolution/Squeeze*
T0*
_class
loc:@conv1d_1/add*
out_type0*
_output_shapes
:
Ї
1training/Adam/gradients/conv1d_1/add_grad/Shape_1Const*
_class
loc:@conv1d_1/add*!
valueB"         *
dtype0*
_output_shapes
:

?training/Adam/gradients/conv1d_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs/training/Adam/gradients/conv1d_1/add_grad/Shape1training/Adam/gradients/conv1d_1/add_grad/Shape_1*
_class
loc:@conv1d_1/add*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

-training/Adam/gradients/conv1d_1/add_grad/SumSum3training/Adam/gradients/conv1d_1/Relu_grad/ReluGrad?training/Adam/gradients/conv1d_1/add_grad/BroadcastGradientArgs*
T0*
_class
loc:@conv1d_1/add*
_output_shapes
:*
	keep_dims( *

Tidx0

1training/Adam/gradients/conv1d_1/add_grad/ReshapeReshape-training/Adam/gradients/conv1d_1/add_grad/Sum/training/Adam/gradients/conv1d_1/add_grad/Shape*
T0*
_class
loc:@conv1d_1/add*
Tshape0*,
_output_shapes
:џџџџџџџџџЌ

/training/Adam/gradients/conv1d_1/add_grad/Sum_1Sum3training/Adam/gradients/conv1d_1/Relu_grad/ReluGradAtraining/Adam/gradients/conv1d_1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_class
loc:@conv1d_1/add*
_output_shapes
:
ў
3training/Adam/gradients/conv1d_1/add_grad/Reshape_1Reshape/training/Adam/gradients/conv1d_1/add_grad/Sum_11training/Adam/gradients/conv1d_1/add_grad/Shape_1*
T0*
_class
loc:@conv1d_1/add*
Tshape0*"
_output_shapes
:
Ы
?training/Adam/gradients/conv1d_1/convolution/Squeeze_grad/ShapeShapeconv1d_1/convolution/Conv2D*
T0*/
_class%
#!loc:@conv1d_1/convolution/Squeeze*
out_type0*
_output_shapes
:
К
Atraining/Adam/gradients/conv1d_1/convolution/Squeeze_grad/ReshapeReshape1training/Adam/gradients/conv1d_1/add_grad/Reshape?training/Adam/gradients/conv1d_1/convolution/Squeeze_grad/Shape*
T0*/
_class%
#!loc:@conv1d_1/convolution/Squeeze*
Tshape0*0
_output_shapes
:џџџџџџџџџЌ
Ђ
3training/Adam/gradients/conv1d_1/Reshape_grad/ShapeConst*
dtype0*
_output_shapes
:*#
_class
loc:@conv1d_1/Reshape*
valueB:

5training/Adam/gradients/conv1d_1/Reshape_grad/ReshapeReshape3training/Adam/gradients/conv1d_1/add_grad/Reshape_13training/Adam/gradients/conv1d_1/Reshape_grad/Shape*
T0*#
_class
loc:@conv1d_1/Reshape*
Tshape0*
_output_shapes
:

?training/Adam/gradients/conv1d_1/convolution/Conv2D_grad/ShapeNShapeNconv1d_1/convolution/ExpandDims!conv1d_1/convolution/ExpandDims_1*
T0*.
_class$
" loc:@conv1d_1/convolution/Conv2D*
out_type0*
N* 
_output_shapes
::
ф
Ltraining/Adam/gradients/conv1d_1/convolution/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput?training/Adam/gradients/conv1d_1/convolution/Conv2D_grad/ShapeN!conv1d_1/convolution/ExpandDims_1Atraining/Adam/gradients/conv1d_1/convolution/Squeeze_grad/Reshape*
paddingVALID*0
_output_shapes
:џџџџџџџџџЏ*
	dilations
*
T0*.
_class$
" loc:@conv1d_1/convolution/Conv2D*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
м
Mtraining/Adam/gradients/conv1d_1/convolution/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterconv1d_1/convolution/ExpandDimsAtraining/Adam/gradients/conv1d_1/convolution/Conv2D_grad/ShapeN:1Atraining/Adam/gradients/conv1d_1/convolution/Squeeze_grad/Reshape*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:*
	dilations
*
T0*.
_class$
" loc:@conv1d_1/convolution/Conv2D
Я
Dtraining/Adam/gradients/conv1d_1/convolution/ExpandDims_1_grad/ShapeConst*
_output_shapes
:*4
_class*
(&loc:@conv1d_1/convolution/ExpandDims_1*!
valueB"         *
dtype0
з
Ftraining/Adam/gradients/conv1d_1/convolution/ExpandDims_1_grad/ReshapeReshapeMtraining/Adam/gradients/conv1d_1/convolution/Conv2D_grad/Conv2DBackpropFilterDtraining/Adam/gradients/conv1d_1/convolution/ExpandDims_1_grad/Shape*
T0*4
_class*
(&loc:@conv1d_1/convolution/ExpandDims_1*
Tshape0*"
_output_shapes
:
_
training/Adam/AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
Ќ
training/Adam/AssignAdd	AssignAddAdam/iterationstraining/Adam/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0	*"
_class
loc:@Adam/iterations
p
training/Adam/CastCastAdam/iterations/read*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0	
X
training/Adam/add/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
b
training/Adam/addAddtraining/Adam/Casttraining/Adam/add/y*
T0*
_output_shapes
: 
^
training/Adam/PowPowAdam/beta_2/readtraining/Adam/add*
T0*
_output_shapes
: 
X
training/Adam/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
T0*
_output_shapes
: 
X
training/Adam/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Z
training/Adam/Const_1Const*
_output_shapes
: *
valueB
 *  *
dtype0
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_1*
T0*
_output_shapes
: 

training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const*
_output_shapes
: *
T0
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
_output_shapes
: *
T0
`
training/Adam/Pow_1PowAdam/beta_1/readtraining/Adam/add*
_output_shapes
: *
T0
Z
training/Adam/sub_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
g
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
_output_shapes
: *
T0
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
T0*
_output_shapes
: 
^
training/Adam/mulMulAdam/lr/readtraining/Adam/truediv*
T0*
_output_shapes
: 
p
training/Adam/zerosConst*!
valueB*    *
dtype0*"
_output_shapes
:

training/Adam/Variable
VariableV2*
shared_name *
dtype0*"
_output_shapes
:*
	container *
shape:
е
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*
validate_shape(*"
_output_shapes
:*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable

training/Adam/Variable/readIdentitytraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*"
_output_shapes
:*
T0
b
training/Adam/zeros_1Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_1
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
е
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:

training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
T0*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes
:
z
%training/Adam/zeros_2/shape_as_tensorConst*
dtype0*
_output_shapes
:*!
valueB"      @   
`
training/Adam/zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
 
training/Adam/zeros_2Fill%training/Adam/zeros_2/shape_as_tensortraining/Adam/zeros_2/Const*
T0*

index_type0*"
_output_shapes
:@

training/Adam/Variable_2
VariableV2*
dtype0*"
_output_shapes
:@*
	container *
shape:@*
shared_name 
н
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*
validate_shape(*"
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2

training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*
T0*+
_class!
loc:@training/Adam/Variable_2*"
_output_shapes
:@
b
training/Adam/zeros_3Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_3
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
е
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/zeros_3*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:@*
use_locking(

training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*
_output_shapes
:@*
T0*+
_class!
loc:@training/Adam/Variable_3
j
training/Adam/zeros_4Const*
dtype0*
_output_shapes

:@*
valueB@*    

training/Adam/Variable_4
VariableV2*
dtype0*
_output_shapes

:@*
	container *
shape
:@*
shared_name 
й
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/zeros_4*
_output_shapes

:@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(

training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*
T0*+
_class!
loc:@training/Adam/Variable_4*
_output_shapes

:@
j
training/Adam/zeros_5Const*
_output_shapes

:*
valueB*    *
dtype0

training/Adam/Variable_5
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 
й
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5

training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
T0*+
_class!
loc:@training/Adam/Variable_5*
_output_shapes

:
b
training/Adam/zeros_6Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_6
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
е
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*
_output_shapes
:

training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
_output_shapes
:*
T0*+
_class!
loc:@training/Adam/Variable_6
v
%training/Adam/zeros_7/shape_as_tensorConst*
valueB"J     *
dtype0*
_output_shapes
:
`
training/Adam/zeros_7/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_7Fill%training/Adam/zeros_7/shape_as_tensortraining/Adam/zeros_7/Const*
T0*

index_type0* 
_output_shapes
:
Ъ

training/Adam/Variable_7
VariableV2*
shape:
Ъ*
shared_name *
dtype0* 
_output_shapes
:
Ъ*
	container 
л
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/zeros_7*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(* 
_output_shapes
:
Ъ

training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7* 
_output_shapes
:
Ъ*
T0*+
_class!
loc:@training/Adam/Variable_7
d
training/Adam/zeros_8Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_8
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
ж
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*
T0*+
_class!
loc:@training/Adam/Variable_8*
_output_shapes	
:
v
%training/Adam/zeros_9/shape_as_tensorConst*
_output_shapes
:*
valueB"      *
dtype0
`
training/Adam/zeros_9/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_9Fill%training/Adam/zeros_9/shape_as_tensortraining/Adam/zeros_9/Const*
T0*

index_type0* 
_output_shapes
:


training/Adam/Variable_9
VariableV2*
dtype0* 
_output_shapes
:
*
	container *
shape:
*
shared_name 
л
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(* 
_output_shapes
:


training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*+
_class!
loc:@training/Adam/Variable_9* 
_output_shapes
:
*
T0
e
training/Adam/zeros_10Const*
dtype0*
_output_shapes	
:*
valueB*    

training/Adam/Variable_10
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
к
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10

training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*
T0*,
_class"
 loc:@training/Adam/Variable_10*
_output_shapes	
:
m
training/Adam/zeros_11Const*
dtype0*
_output_shapes
:	*
valueB	*    

training/Adam/Variable_11
VariableV2*
dtype0*
_output_shapes
:	*
	container *
shape:	*
shared_name 
о
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/zeros_11*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:	

training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*
T0*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
:	
c
training/Adam/zeros_12Const*
_output_shapes
:*
valueB*    *
dtype0

training/Adam/Variable_12
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
й
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*
_output_shapes
:

training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_12
s
training/Adam/zeros_13Const*!
valueB*    *
dtype0*"
_output_shapes
:

training/Adam/Variable_13
VariableV2*
shape:*
shared_name *
dtype0*"
_output_shapes
:*
	container 
с
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*"
_output_shapes
:
 
training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*"
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_13
c
training/Adam/zeros_14Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_14
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
й
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*
_output_shapes
:

training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
T0*,
_class"
 loc:@training/Adam/Variable_14*
_output_shapes
:
{
&training/Adam/zeros_15/shape_as_tensorConst*!
valueB"      @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_15/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ѓ
training/Adam/zeros_15Fill&training/Adam/zeros_15/shape_as_tensortraining/Adam/zeros_15/Const*
T0*

index_type0*"
_output_shapes
:@

training/Adam/Variable_15
VariableV2*
shared_name *
dtype0*"
_output_shapes
:@*
	container *
shape:@
с
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/zeros_15*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*"
_output_shapes
:@
 
training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*
T0*,
_class"
 loc:@training/Adam/Variable_15*"
_output_shapes
:@
c
training/Adam/zeros_16Const*
_output_shapes
:@*
valueB@*    *
dtype0

training/Adam/Variable_16
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
й
 training/Adam/Variable_16/AssignAssigntraining/Adam/Variable_16training/Adam/zeros_16*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_16/readIdentitytraining/Adam/Variable_16*
T0*,
_class"
 loc:@training/Adam/Variable_16*
_output_shapes
:@
k
training/Adam/zeros_17Const*
valueB@*    *
dtype0*
_output_shapes

:@

training/Adam/Variable_17
VariableV2*
dtype0*
_output_shapes

:@*
	container *
shape
:@*
shared_name 
н
 training/Adam/Variable_17/AssignAssigntraining/Adam/Variable_17training/Adam/zeros_17*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes

:@*
use_locking(

training/Adam/Variable_17/readIdentitytraining/Adam/Variable_17*,
_class"
 loc:@training/Adam/Variable_17*
_output_shapes

:@*
T0
k
training/Adam/zeros_18Const*
valueB*    *
dtype0*
_output_shapes

:

training/Adam/Variable_18
VariableV2*
shared_name *
dtype0*
_output_shapes

:*
	container *
shape
:
н
 training/Adam/Variable_18/AssignAssigntraining/Adam/Variable_18training/Adam/zeros_18*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*
_output_shapes

:*
use_locking(

training/Adam/Variable_18/readIdentitytraining/Adam/Variable_18*
T0*,
_class"
 loc:@training/Adam/Variable_18*
_output_shapes

:
c
training/Adam/zeros_19Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_19
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
й
 training/Adam/Variable_19/AssignAssigntraining/Adam/Variable_19training/Adam/zeros_19*
T0*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_19/readIdentitytraining/Adam/Variable_19*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_19
w
&training/Adam/zeros_20/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"J     
a
training/Adam/zeros_20/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ё
training/Adam/zeros_20Fill&training/Adam/zeros_20/shape_as_tensortraining/Adam/zeros_20/Const*
T0*

index_type0* 
_output_shapes
:
Ъ

training/Adam/Variable_20
VariableV2*
shape:
Ъ*
shared_name *
dtype0* 
_output_shapes
:
Ъ*
	container 
п
 training/Adam/Variable_20/AssignAssigntraining/Adam/Variable_20training/Adam/zeros_20*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20*
validate_shape(* 
_output_shapes
:
Ъ

training/Adam/Variable_20/readIdentitytraining/Adam/Variable_20*
T0*,
_class"
 loc:@training/Adam/Variable_20* 
_output_shapes
:
Ъ
e
training/Adam/zeros_21Const*
dtype0*
_output_shapes	
:*
valueB*    

training/Adam/Variable_21
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
к
 training/Adam/Variable_21/AssignAssigntraining/Adam/Variable_21training/Adam/zeros_21*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_21*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_21/readIdentitytraining/Adam/Variable_21*
_output_shapes	
:*
T0*,
_class"
 loc:@training/Adam/Variable_21
w
&training/Adam/zeros_22/shape_as_tensorConst*
_output_shapes
:*
valueB"      *
dtype0
a
training/Adam/zeros_22/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ё
training/Adam/zeros_22Fill&training/Adam/zeros_22/shape_as_tensortraining/Adam/zeros_22/Const*
T0*

index_type0* 
_output_shapes
:


training/Adam/Variable_22
VariableV2*
shape:
*
shared_name *
dtype0* 
_output_shapes
:
*
	container 
п
 training/Adam/Variable_22/AssignAssigntraining/Adam/Variable_22training/Adam/zeros_22*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(* 
_output_shapes
:


training/Adam/Variable_22/readIdentitytraining/Adam/Variable_22* 
_output_shapes
:
*
T0*,
_class"
 loc:@training/Adam/Variable_22
e
training/Adam/zeros_23Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_23
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
к
 training/Adam/Variable_23/AssignAssigntraining/Adam/Variable_23training/Adam/zeros_23*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_23

training/Adam/Variable_23/readIdentitytraining/Adam/Variable_23*
T0*,
_class"
 loc:@training/Adam/Variable_23*
_output_shapes	
:
m
training/Adam/zeros_24Const*
valueB	*    *
dtype0*
_output_shapes
:	

training/Adam/Variable_24
VariableV2*
shared_name *
dtype0*
_output_shapes
:	*
	container *
shape:	
о
 training/Adam/Variable_24/AssignAssigntraining/Adam/Variable_24training/Adam/zeros_24*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_24

training/Adam/Variable_24/readIdentitytraining/Adam/Variable_24*
T0*,
_class"
 loc:@training/Adam/Variable_24*
_output_shapes
:	
c
training/Adam/zeros_25Const*
_output_shapes
:*
valueB*    *
dtype0

training/Adam/Variable_25
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
й
 training/Adam/Variable_25/AssignAssigntraining/Adam/Variable_25training/Adam/zeros_25*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_25*
validate_shape(

training/Adam/Variable_25/readIdentitytraining/Adam/Variable_25*
T0*,
_class"
 loc:@training/Adam/Variable_25*
_output_shapes
:
p
&training/Adam/zeros_26/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_26/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_26Fill&training/Adam/zeros_26/shape_as_tensortraining/Adam/zeros_26/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_26
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
й
 training/Adam/Variable_26/AssignAssigntraining/Adam/Variable_26training/Adam/zeros_26*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_26*
validate_shape(*
_output_shapes
:

training/Adam/Variable_26/readIdentitytraining/Adam/Variable_26*
T0*,
_class"
 loc:@training/Adam/Variable_26*
_output_shapes
:
p
&training/Adam/zeros_27/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_27/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_27Fill&training/Adam/zeros_27/shape_as_tensortraining/Adam/zeros_27/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_27
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
й
 training/Adam/Variable_27/AssignAssigntraining/Adam/Variable_27training/Adam/zeros_27*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_27

training/Adam/Variable_27/readIdentitytraining/Adam/Variable_27*,
_class"
 loc:@training/Adam/Variable_27*
_output_shapes
:*
T0
p
&training/Adam/zeros_28/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_28/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_28Fill&training/Adam/zeros_28/shape_as_tensortraining/Adam/zeros_28/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_28
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
й
 training/Adam/Variable_28/AssignAssigntraining/Adam/Variable_28training/Adam/zeros_28*
T0*,
_class"
 loc:@training/Adam/Variable_28*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_28/readIdentitytraining/Adam/Variable_28*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_28
p
&training/Adam/zeros_29/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_29/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_29Fill&training/Adam/zeros_29/shape_as_tensortraining/Adam/zeros_29/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_29
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
й
 training/Adam/Variable_29/AssignAssigntraining/Adam/Variable_29training/Adam/zeros_29*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_29*
validate_shape(*
_output_shapes
:

training/Adam/Variable_29/readIdentitytraining/Adam/Variable_29*
T0*,
_class"
 loc:@training/Adam/Variable_29*
_output_shapes
:
p
&training/Adam/zeros_30/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_30/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_30Fill&training/Adam/zeros_30/shape_as_tensortraining/Adam/zeros_30/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_30
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
й
 training/Adam/Variable_30/AssignAssigntraining/Adam/Variable_30training/Adam/zeros_30*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_30*
validate_shape(

training/Adam/Variable_30/readIdentitytraining/Adam/Variable_30*
T0*,
_class"
 loc:@training/Adam/Variable_30*
_output_shapes
:
p
&training/Adam/zeros_31/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_31/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_31Fill&training/Adam/zeros_31/shape_as_tensortraining/Adam/zeros_31/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_31
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
й
 training/Adam/Variable_31/AssignAssigntraining/Adam/Variable_31training/Adam/zeros_31*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_31*
validate_shape(*
_output_shapes
:

training/Adam/Variable_31/readIdentitytraining/Adam/Variable_31*
T0*,
_class"
 loc:@training/Adam/Variable_31*
_output_shapes
:
p
&training/Adam/zeros_32/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_32/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_32Fill&training/Adam/zeros_32/shape_as_tensortraining/Adam/zeros_32/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_32
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
й
 training/Adam/Variable_32/AssignAssigntraining/Adam/Variable_32training/Adam/zeros_32*,
_class"
 loc:@training/Adam/Variable_32*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_32/readIdentitytraining/Adam/Variable_32*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_32
p
&training/Adam/zeros_33/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_33/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_33Fill&training/Adam/zeros_33/shape_as_tensortraining/Adam/zeros_33/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_33
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
й
 training/Adam/Variable_33/AssignAssigntraining/Adam/Variable_33training/Adam/zeros_33*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_33*
validate_shape(

training/Adam/Variable_33/readIdentitytraining/Adam/Variable_33*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_33
p
&training/Adam/zeros_34/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_34/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_34Fill&training/Adam/zeros_34/shape_as_tensortraining/Adam/zeros_34/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_34
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
й
 training/Adam/Variable_34/AssignAssigntraining/Adam/Variable_34training/Adam/zeros_34*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_34*
validate_shape(

training/Adam/Variable_34/readIdentitytraining/Adam/Variable_34*
T0*,
_class"
 loc:@training/Adam/Variable_34*
_output_shapes
:
p
&training/Adam/zeros_35/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_35/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_35Fill&training/Adam/zeros_35/shape_as_tensortraining/Adam/zeros_35/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_35
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
й
 training/Adam/Variable_35/AssignAssigntraining/Adam/Variable_35training/Adam/zeros_35*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_35*
validate_shape(*
_output_shapes
:

training/Adam/Variable_35/readIdentitytraining/Adam/Variable_35*
T0*,
_class"
 loc:@training/Adam/Variable_35*
_output_shapes
:
p
&training/Adam/zeros_36/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_36/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_36Fill&training/Adam/zeros_36/shape_as_tensortraining/Adam/zeros_36/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_36
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
й
 training/Adam/Variable_36/AssignAssigntraining/Adam/Variable_36training/Adam/zeros_36*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_36*
validate_shape(*
_output_shapes
:

training/Adam/Variable_36/readIdentitytraining/Adam/Variable_36*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_36
p
&training/Adam/zeros_37/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_37/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_37Fill&training/Adam/zeros_37/shape_as_tensortraining/Adam/zeros_37/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_37
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
й
 training/Adam/Variable_37/AssignAssigntraining/Adam/Variable_37training/Adam/zeros_37*,
_class"
 loc:@training/Adam/Variable_37*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_37/readIdentitytraining/Adam/Variable_37*,
_class"
 loc:@training/Adam/Variable_37*
_output_shapes
:*
T0
p
&training/Adam/zeros_38/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_38/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_38Fill&training/Adam/zeros_38/shape_as_tensortraining/Adam/zeros_38/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_38
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
й
 training/Adam/Variable_38/AssignAssigntraining/Adam/Variable_38training/Adam/zeros_38*
T0*,
_class"
 loc:@training/Adam/Variable_38*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_38/readIdentitytraining/Adam/Variable_38*
T0*,
_class"
 loc:@training/Adam/Variable_38*
_output_shapes
:
v
training/Adam/mul_1MulAdam/beta_1/readtraining/Adam/Variable/read*"
_output_shapes
:*
T0
Z
training/Adam/sub_2/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
d
training/Adam/sub_2Subtraining/Adam/sub_2/xAdam/beta_1/read*
T0*
_output_shapes
: 
Є
training/Adam/mul_2Multraining/Adam/sub_2Ftraining/Adam/gradients/conv1d_1/convolution/ExpandDims_1_grad/Reshape*
T0*"
_output_shapes
:
q
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*"
_output_shapes
:*
T0
y
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_13/read*"
_output_shapes
:*
T0
Z
training/Adam/sub_3/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_3Subtraining/Adam/sub_3/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/SquareSquareFtraining/Adam/gradients/conv1d_1/convolution/ExpandDims_1_grad/Reshape*"
_output_shapes
:*
T0
r
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
T0*"
_output_shapes
:
q
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*"
_output_shapes
:*
T0
o
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
T0*"
_output_shapes
:
Z
training/Adam/Const_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_3Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_3*"
_output_shapes
:*
T0

training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_2*
T0*"
_output_shapes
:
h
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0*"
_output_shapes
:
Z
training/Adam/add_3/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
t
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*"
_output_shapes
:*
T0
y
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
T0*"
_output_shapes
:
v
training/Adam/sub_4Subconv1d_1/kernel/readtraining/Adam/truediv_1*"
_output_shapes
:*
T0
Ь
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*
validate_shape(*"
_output_shapes
:*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable
д
training/Adam/Assign_1Assigntraining/Adam/Variable_13training/Adam/add_2*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*"
_output_shapes
:
Р
training/Adam/Assign_2Assignconv1d_1/kerneltraining/Adam/sub_4*"
_class
loc:@conv1d_1/kernel*
validate_shape(*"
_output_shapes
:*
use_locking(*
T0
p
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
T0*
_output_shapes
:
Z
training/Adam/sub_5/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_5Subtraining/Adam/sub_5/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_7Multraining/Adam/sub_55training/Adam/gradients/conv1d_1/Reshape_grad/Reshape*
T0*
_output_shapes
:
i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
T0*
_output_shapes
:
q
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_14/read*
_output_shapes
:*
T0
Z
training/Adam/sub_6/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_6Subtraining/Adam/sub_6/xAdam/beta_2/read*
T0*
_output_shapes
: 
|
training/Adam/Square_1Square5training/Adam/gradients/conv1d_1/Reshape_grad/Reshape*
T0*
_output_shapes
:
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
_output_shapes
:*
T0
i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
T0*
_output_shapes
:
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes
:
Z
training/Adam/Const_4Const*
dtype0*
_output_shapes
: *
valueB
 *    
Z
training/Adam/Const_5Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_5*
_output_shapes
:*
T0

training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_4*
T0*
_output_shapes
:
`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
_output_shapes
:*
T0
Z
training/Adam/add_6/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
l
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
T0*
_output_shapes
:
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
T0*
_output_shapes
:
l
training/Adam/sub_7Subconv1d_1/bias/readtraining/Adam/truediv_2*
T0*
_output_shapes
:
Ъ
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:
Ь
training/Adam/Assign_4Assigntraining/Adam/Variable_14training/Adam/add_5*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*
_output_shapes
:
Д
training/Adam/Assign_5Assignconv1d_1/biastraining/Adam/sub_7*
use_locking(*
T0* 
_class
loc:@conv1d_1/bias*
validate_shape(*
_output_shapes
:
y
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*"
_output_shapes
:@*
T0
Z
training/Adam/sub_8/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_8Subtraining/Adam/sub_8/xAdam/beta_1/read*
_output_shapes
: *
T0
Ѕ
training/Adam/mul_12Multraining/Adam/sub_8Ftraining/Adam/gradients/conv1d_2/convolution/ExpandDims_1_grad/Reshape*"
_output_shapes
:@*
T0
s
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
T0*"
_output_shapes
:@
z
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_15/read*"
_output_shapes
:@*
T0
Z
training/Adam/sub_9/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
d
training/Adam/sub_9Subtraining/Adam/sub_9/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_2SquareFtraining/Adam/gradients/conv1d_2/convolution/ExpandDims_1_grad/Reshape*
T0*"
_output_shapes
:@
u
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*"
_output_shapes
:@
s
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
T0*"
_output_shapes
:@
p
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*
T0*"
_output_shapes
:@
Z
training/Adam/Const_6Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_7Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_7*
T0*"
_output_shapes
:@

training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_6*"
_output_shapes
:@*
T0
h
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0*"
_output_shapes
:@
Z
training/Adam/add_9/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
t
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y*
T0*"
_output_shapes
:@
z
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
T0*"
_output_shapes
:@
w
training/Adam/sub_10Subconv1d_2/kernel/readtraining/Adam/truediv_3*"
_output_shapes
:@*
T0
в
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*"
_output_shapes
:@*
use_locking(*
T0
д
training/Adam/Assign_7Assigntraining/Adam/Variable_15training/Adam/add_8*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*"
_output_shapes
:@*
use_locking(*
T0
С
training/Adam/Assign_8Assignconv1d_2/kerneltraining/Adam/sub_10*
use_locking(*
T0*"
_class
loc:@conv1d_2/kernel*
validate_shape(*"
_output_shapes
:@
q
training/Adam/mul_16MulAdam/beta_1/readtraining/Adam/Variable_3/read*
T0*
_output_shapes
:@
[
training/Adam/sub_11/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_11Subtraining/Adam/sub_11/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_17Multraining/Adam/sub_115training/Adam/gradients/conv1d_2/Reshape_grad/Reshape*
T0*
_output_shapes
:@
l
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
T0*
_output_shapes
:@
r
training/Adam/mul_18MulAdam/beta_2/readtraining/Adam/Variable_16/read*
_output_shapes
:@*
T0
[
training/Adam/sub_12/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_12Subtraining/Adam/sub_12/xAdam/beta_2/read*
T0*
_output_shapes
: 
|
training/Adam/Square_3Square5training/Adam/gradients/conv1d_2/Reshape_grad/Reshape*
_output_shapes
:@*
T0
n
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
T0*
_output_shapes
:@
l
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
T0*
_output_shapes
:@
i
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
T0*
_output_shapes
:@
Z
training/Adam/Const_8Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_9Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_9*
_output_shapes
:@*
T0

training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_8*
T0*
_output_shapes
:@
`
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
T0*
_output_shapes
:@
[
training/Adam/add_12/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
n
training/Adam/add_12Addtraining/Adam/Sqrt_4training/Adam/add_12/y*
T0*
_output_shapes
:@
s
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
T0*
_output_shapes
:@
m
training/Adam/sub_13Subconv1d_2/bias/readtraining/Adam/truediv_4*
T0*
_output_shapes
:@
Ы
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:@
Ю
training/Adam/Assign_10Assigntraining/Adam/Variable_16training/Adam/add_11*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(
Ж
training/Adam/Assign_11Assignconv1d_2/biastraining/Adam/sub_13*
use_locking(*
T0* 
_class
loc:@conv1d_2/bias*
validate_shape(*
_output_shapes
:@
u
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*
_output_shapes

:@*
T0
[
training/Adam/sub_14/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_14Subtraining/Adam/sub_14/xAdam/beta_1/read*
_output_shapes
: *
T0
{
training/Adam/mul_22Multraining/Adam/sub_14training/Adam/gradients/AddN_11*
_output_shapes

:@*
T0
p
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
T0*
_output_shapes

:@
v
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_17/read*
_output_shapes

:@*
T0
[
training/Adam/sub_15/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_15Subtraining/Adam/sub_15/xAdam/beta_2/read*
T0*
_output_shapes
: 
j
training/Adam/Square_4Squaretraining/Adam/gradients/AddN_11*
_output_shapes

:@*
T0
r
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
T0*
_output_shapes

:@
p
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
T0*
_output_shapes

:@
m
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
_output_shapes

:@*
T0
[
training/Adam/Const_10Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_11Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_11*
T0*
_output_shapes

:@

training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_10*
_output_shapes

:@*
T0
d
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
_output_shapes

:@*
T0
[
training/Adam/add_15/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
r
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y*
T0*
_output_shapes

:@
w
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
_output_shapes

:@*
T0
q
training/Adam/sub_16Sublstm_1/kernel/readtraining/Adam/truediv_5*
T0*
_output_shapes

:@
а
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
_output_shapes

:@
в
training/Adam/Assign_13Assigntraining/Adam/Variable_17training/Adam/add_14*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
К
training/Adam/Assign_14Assignlstm_1/kerneltraining/Adam/sub_16*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0* 
_class
loc:@lstm_1/kernel
u
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read*
T0*
_output_shapes

:
[
training/Adam/sub_17/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_17Subtraining/Adam/sub_17/xAdam/beta_1/read*
T0*
_output_shapes
: 
{
training/Adam/mul_27Multraining/Adam/sub_17training/Adam/gradients/AddN_10*
_output_shapes

:*
T0
p
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
T0*
_output_shapes

:
v
training/Adam/mul_28MulAdam/beta_2/readtraining/Adam/Variable_18/read*
_output_shapes

:*
T0
[
training/Adam/sub_18/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_18Subtraining/Adam/sub_18/xAdam/beta_2/read*
T0*
_output_shapes
: 
j
training/Adam/Square_5Squaretraining/Adam/gradients/AddN_10*
_output_shapes

:*
T0
r
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
T0*
_output_shapes

:
p
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
_output_shapes

:*
T0
m
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
T0*
_output_shapes

:
[
training/Adam/Const_12Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_13Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_13*
_output_shapes

:*
T0

training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_12*
T0*
_output_shapes

:
d
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
_output_shapes

:*
T0
[
training/Adam/add_18/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
r
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y*
_output_shapes

:*
T0
w
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
T0*
_output_shapes

:
{
training/Adam/sub_19Sublstm_1/recurrent_kernel/readtraining/Adam/truediv_6*
T0*
_output_shapes

:
а
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes

:*
use_locking(
в
training/Adam/Assign_16Assigntraining/Adam/Variable_18training/Adam/add_17*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(
Ю
training/Adam/Assign_17Assignlstm_1/recurrent_kerneltraining/Adam/sub_19*
validate_shape(*
_output_shapes

:*
use_locking(*
T0**
_class 
loc:@lstm_1/recurrent_kernel
q
training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*
_output_shapes
:*
T0
[
training/Adam/sub_20/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_20Subtraining/Adam/sub_20/xAdam/beta_1/read*
T0*
_output_shapes
: 
v
training/Adam/mul_32Multraining/Adam/sub_20training/Adam/gradients/AddN_9*
T0*
_output_shapes
:
l
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
_output_shapes
:*
T0
r
training/Adam/mul_33MulAdam/beta_2/readtraining/Adam/Variable_19/read*
_output_shapes
:*
T0
[
training/Adam/sub_21/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_21Subtraining/Adam/sub_21/xAdam/beta_2/read*
_output_shapes
: *
T0
e
training/Adam/Square_6Squaretraining/Adam/gradients/AddN_9*
T0*
_output_shapes
:
n
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
T0*
_output_shapes
:
l
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
T0*
_output_shapes
:
i
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
T0*
_output_shapes
:
[
training/Adam/Const_14Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_15Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_15*
_output_shapes
:*
T0

training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_14*
T0*
_output_shapes
:
`
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
_output_shapes
:*
T0
[
training/Adam/add_21/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
n
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*
T0*
_output_shapes
:
s
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
T0*
_output_shapes
:
k
training/Adam/sub_22Sublstm_1/bias/readtraining/Adam/truediv_7*
T0*
_output_shapes
:
Ь
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(
Ю
training/Adam/Assign_19Assigntraining/Adam/Variable_19training/Adam/add_20*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes
:
В
training/Adam/Assign_20Assignlstm_1/biastraining/Adam/sub_22*
_class
loc:@lstm_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
w
training/Adam/mul_36MulAdam/beta_1/readtraining/Adam/Variable_7/read*
T0* 
_output_shapes
:
Ъ
[
training/Adam/sub_23/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_23Subtraining/Adam/sub_23/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_37Multraining/Adam/sub_234training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
Ъ
r
training/Adam/add_22Addtraining/Adam/mul_36training/Adam/mul_37* 
_output_shapes
:
Ъ*
T0
x
training/Adam/mul_38MulAdam/beta_2/readtraining/Adam/Variable_20/read* 
_output_shapes
:
Ъ*
T0
[
training/Adam/sub_24/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_24Subtraining/Adam/sub_24/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_7Square4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
Ъ
t
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7*
T0* 
_output_shapes
:
Ъ
r
training/Adam/add_23Addtraining/Adam/mul_38training/Adam/mul_39*
T0* 
_output_shapes
:
Ъ
o
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22* 
_output_shapes
:
Ъ*
T0
[
training/Adam/Const_16Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_17Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_17*
T0* 
_output_shapes
:
Ъ

training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_16*
T0* 
_output_shapes
:
Ъ
f
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
T0* 
_output_shapes
:
Ъ
[
training/Adam/add_24/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
t
training/Adam/add_24Addtraining/Adam/Sqrt_8training/Adam/add_24/y* 
_output_shapes
:
Ъ*
T0
y
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24* 
_output_shapes
:
Ъ*
T0
t
training/Adam/sub_25Subdense_1/kernel/readtraining/Adam/truediv_8*
T0* 
_output_shapes
:
Ъ
в
training/Adam/Assign_21Assigntraining/Adam/Variable_7training/Adam/add_22*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(* 
_output_shapes
:
Ъ
д
training/Adam/Assign_22Assigntraining/Adam/Variable_20training/Adam/add_23* 
_output_shapes
:
Ъ*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20*
validate_shape(
О
training/Adam/Assign_23Assigndense_1/kerneltraining/Adam/sub_25*
validate_shape(* 
_output_shapes
:
Ъ*
use_locking(*
T0*!
_class
loc:@dense_1/kernel
r
training/Adam/mul_41MulAdam/beta_1/readtraining/Adam/Variable_8/read*
_output_shapes	
:*
T0
[
training/Adam/sub_26/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_26Subtraining/Adam/sub_26/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_42Multraining/Adam/sub_268training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
m
training/Adam/add_25Addtraining/Adam/mul_41training/Adam/mul_42*
T0*
_output_shapes	
:
s
training/Adam/mul_43MulAdam/beta_2/readtraining/Adam/Variable_21/read*
T0*
_output_shapes	
:
[
training/Adam/sub_27/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_27Subtraining/Adam/sub_27/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_8Square8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
o
training/Adam/mul_44Multraining/Adam/sub_27training/Adam/Square_8*
T0*
_output_shapes	
:
m
training/Adam/add_26Addtraining/Adam/mul_43training/Adam/mul_44*
T0*
_output_shapes	
:
j
training/Adam/mul_45Multraining/Adam/multraining/Adam/add_25*
T0*
_output_shapes	
:
[
training/Adam/Const_18Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_19Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_9/MinimumMinimumtraining/Adam/add_26training/Adam/Const_19*
T0*
_output_shapes	
:

training/Adam/clip_by_value_9Maximum%training/Adam/clip_by_value_9/Minimumtraining/Adam/Const_18*
T0*
_output_shapes	
:
a
training/Adam/Sqrt_9Sqrttraining/Adam/clip_by_value_9*
_output_shapes	
:*
T0
[
training/Adam/add_27/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
o
training/Adam/add_27Addtraining/Adam/Sqrt_9training/Adam/add_27/y*
T0*
_output_shapes	
:
t
training/Adam/truediv_9RealDivtraining/Adam/mul_45training/Adam/add_27*
_output_shapes	
:*
T0
m
training/Adam/sub_28Subdense_1/bias/readtraining/Adam/truediv_9*
_output_shapes	
:*
T0
Э
training/Adam/Assign_24Assigntraining/Adam/Variable_8training/Adam/add_25*
_output_shapes	
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(
Я
training/Adam/Assign_25Assigntraining/Adam/Variable_21training/Adam/add_26*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_21*
validate_shape(*
_output_shapes	
:
Е
training/Adam/Assign_26Assigndense_1/biastraining/Adam/sub_28*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
w
training/Adam/mul_46MulAdam/beta_1/readtraining/Adam/Variable_9/read*
T0* 
_output_shapes
:

[
training/Adam/sub_29/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_29Subtraining/Adam/sub_29/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_47Multraining/Adam/sub_294training/Adam/gradients/dense_2/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
r
training/Adam/add_28Addtraining/Adam/mul_46training/Adam/mul_47*
T0* 
_output_shapes
:

x
training/Adam/mul_48MulAdam/beta_2/readtraining/Adam/Variable_22/read*
T0* 
_output_shapes
:

[
training/Adam/sub_30/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_30Subtraining/Adam/sub_30/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_9Square4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

t
training/Adam/mul_49Multraining/Adam/sub_30training/Adam/Square_9*
T0* 
_output_shapes
:

r
training/Adam/add_29Addtraining/Adam/mul_48training/Adam/mul_49*
T0* 
_output_shapes
:

o
training/Adam/mul_50Multraining/Adam/multraining/Adam/add_28* 
_output_shapes
:
*
T0
[
training/Adam/Const_20Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_21Const*
dtype0*
_output_shapes
: *
valueB
 *  

&training/Adam/clip_by_value_10/MinimumMinimumtraining/Adam/add_29training/Adam/Const_21*
T0* 
_output_shapes
:


training/Adam/clip_by_value_10Maximum&training/Adam/clip_by_value_10/Minimumtraining/Adam/Const_20* 
_output_shapes
:
*
T0
h
training/Adam/Sqrt_10Sqrttraining/Adam/clip_by_value_10*
T0* 
_output_shapes
:

[
training/Adam/add_30/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
u
training/Adam/add_30Addtraining/Adam/Sqrt_10training/Adam/add_30/y* 
_output_shapes
:
*
T0
z
training/Adam/truediv_10RealDivtraining/Adam/mul_50training/Adam/add_30*
T0* 
_output_shapes
:

u
training/Adam/sub_31Subdense_2/kernel/readtraining/Adam/truediv_10*
T0* 
_output_shapes
:

в
training/Adam/Assign_27Assigntraining/Adam/Variable_9training/Adam/add_28*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(* 
_output_shapes
:

д
training/Adam/Assign_28Assigntraining/Adam/Variable_22training/Adam/add_29*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(* 
_output_shapes
:

О
training/Adam/Assign_29Assigndense_2/kerneltraining/Adam/sub_31*!
_class
loc:@dense_2/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
s
training/Adam/mul_51MulAdam/beta_1/readtraining/Adam/Variable_10/read*
T0*
_output_shapes	
:
[
training/Adam/sub_32/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_32Subtraining/Adam/sub_32/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_52Multraining/Adam/sub_328training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
m
training/Adam/add_31Addtraining/Adam/mul_51training/Adam/mul_52*
_output_shapes	
:*
T0
s
training/Adam/mul_53MulAdam/beta_2/readtraining/Adam/Variable_23/read*
T0*
_output_shapes	
:
[
training/Adam/sub_33/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_33Subtraining/Adam/sub_33/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_10Square8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
p
training/Adam/mul_54Multraining/Adam/sub_33training/Adam/Square_10*
T0*
_output_shapes	
:
m
training/Adam/add_32Addtraining/Adam/mul_53training/Adam/mul_54*
T0*
_output_shapes	
:
j
training/Adam/mul_55Multraining/Adam/multraining/Adam/add_31*
T0*
_output_shapes	
:
[
training/Adam/Const_22Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_23Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_11/MinimumMinimumtraining/Adam/add_32training/Adam/Const_23*
_output_shapes	
:*
T0

training/Adam/clip_by_value_11Maximum&training/Adam/clip_by_value_11/Minimumtraining/Adam/Const_22*
_output_shapes	
:*
T0
c
training/Adam/Sqrt_11Sqrttraining/Adam/clip_by_value_11*
_output_shapes	
:*
T0
[
training/Adam/add_33/yConst*
dtype0*
_output_shapes
: *
valueB
 *Пж3
p
training/Adam/add_33Addtraining/Adam/Sqrt_11training/Adam/add_33/y*
T0*
_output_shapes	
:
u
training/Adam/truediv_11RealDivtraining/Adam/mul_55training/Adam/add_33*
T0*
_output_shapes	
:
n
training/Adam/sub_34Subdense_2/bias/readtraining/Adam/truediv_11*
T0*
_output_shapes	
:
Я
training/Adam/Assign_30Assigntraining/Adam/Variable_10training/Adam/add_31*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(
Я
training/Adam/Assign_31Assigntraining/Adam/Variable_23training/Adam/add_32*,
_class"
 loc:@training/Adam/Variable_23*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Е
training/Adam/Assign_32Assigndense_2/biastraining/Adam/sub_34*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes	
:
w
training/Adam/mul_56MulAdam/beta_1/readtraining/Adam/Variable_11/read*
_output_shapes
:	*
T0
[
training/Adam/sub_35/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_35Subtraining/Adam/sub_35/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_57Multraining/Adam/sub_354training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
_output_shapes
:	*
T0
q
training/Adam/add_34Addtraining/Adam/mul_56training/Adam/mul_57*
T0*
_output_shapes
:	
w
training/Adam/mul_58MulAdam/beta_2/readtraining/Adam/Variable_24/read*
T0*
_output_shapes
:	
[
training/Adam/sub_36/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_36Subtraining/Adam/sub_36/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_11Square4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	
t
training/Adam/mul_59Multraining/Adam/sub_36training/Adam/Square_11*
T0*
_output_shapes
:	
q
training/Adam/add_35Addtraining/Adam/mul_58training/Adam/mul_59*
T0*
_output_shapes
:	
n
training/Adam/mul_60Multraining/Adam/multraining/Adam/add_34*
T0*
_output_shapes
:	
[
training/Adam/Const_24Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_25Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_12/MinimumMinimumtraining/Adam/add_35training/Adam/Const_25*
T0*
_output_shapes
:	

training/Adam/clip_by_value_12Maximum&training/Adam/clip_by_value_12/Minimumtraining/Adam/Const_24*
T0*
_output_shapes
:	
g
training/Adam/Sqrt_12Sqrttraining/Adam/clip_by_value_12*
_output_shapes
:	*
T0
[
training/Adam/add_36/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
t
training/Adam/add_36Addtraining/Adam/Sqrt_12training/Adam/add_36/y*
T0*
_output_shapes
:	
y
training/Adam/truediv_12RealDivtraining/Adam/mul_60training/Adam/add_36*
T0*
_output_shapes
:	
t
training/Adam/sub_37Subdense_3/kernel/readtraining/Adam/truediv_12*
_output_shapes
:	*
T0
г
training/Adam/Assign_33Assigntraining/Adam/Variable_11training/Adam/add_34*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:	
г
training/Adam/Assign_34Assigntraining/Adam/Variable_24training/Adam/add_35*
_output_shapes
:	*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_24*
validate_shape(
Н
training/Adam/Assign_35Assigndense_3/kerneltraining/Adam/sub_37*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes
:	
r
training/Adam/mul_61MulAdam/beta_1/readtraining/Adam/Variable_12/read*
_output_shapes
:*
T0
[
training/Adam/sub_38/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_38Subtraining/Adam/sub_38/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_62Multraining/Adam/sub_388training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
l
training/Adam/add_37Addtraining/Adam/mul_61training/Adam/mul_62*
T0*
_output_shapes
:
r
training/Adam/mul_63MulAdam/beta_2/readtraining/Adam/Variable_25/read*
_output_shapes
:*
T0
[
training/Adam/sub_39/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_39Subtraining/Adam/sub_39/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_12Square8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
o
training/Adam/mul_64Multraining/Adam/sub_39training/Adam/Square_12*
T0*
_output_shapes
:
l
training/Adam/add_38Addtraining/Adam/mul_63training/Adam/mul_64*
_output_shapes
:*
T0
i
training/Adam/mul_65Multraining/Adam/multraining/Adam/add_37*
_output_shapes
:*
T0
[
training/Adam/Const_26Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_27Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_13/MinimumMinimumtraining/Adam/add_38training/Adam/Const_27*
T0*
_output_shapes
:

training/Adam/clip_by_value_13Maximum&training/Adam/clip_by_value_13/Minimumtraining/Adam/Const_26*
_output_shapes
:*
T0
b
training/Adam/Sqrt_13Sqrttraining/Adam/clip_by_value_13*
T0*
_output_shapes
:
[
training/Adam/add_39/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
o
training/Adam/add_39Addtraining/Adam/Sqrt_13training/Adam/add_39/y*
_output_shapes
:*
T0
t
training/Adam/truediv_13RealDivtraining/Adam/mul_65training/Adam/add_39*
T0*
_output_shapes
:
m
training/Adam/sub_40Subdense_3/bias/readtraining/Adam/truediv_13*
_output_shapes
:*
T0
Ю
training/Adam/Assign_36Assigntraining/Adam/Variable_12training/Adam/add_37*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*
_output_shapes
:
Ю
training/Adam/Assign_37Assigntraining/Adam/Variable_25training/Adam/add_38*,
_class"
 loc:@training/Adam/Variable_25*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Д
training/Adam/Assign_38Assigndense_3/biastraining/Adam/sub_40*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@dense_3/bias
Н
training/group_depsNoOp	^loss/mul^metrics/acc/Mean^training/Adam/Assign^training/Adam/AssignAdd^training/Adam/Assign_1^training/Adam/Assign_10^training/Adam/Assign_11^training/Adam/Assign_12^training/Adam/Assign_13^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_18^training/Adam/Assign_19^training/Adam/Assign_2^training/Adam/Assign_20^training/Adam/Assign_21^training/Adam/Assign_22^training/Adam/Assign_23^training/Adam/Assign_24^training/Adam/Assign_25^training/Adam/Assign_26^training/Adam/Assign_27^training/Adam/Assign_28^training/Adam/Assign_29^training/Adam/Assign_3^training/Adam/Assign_30^training/Adam/Assign_31^training/Adam/Assign_32^training/Adam/Assign_33^training/Adam/Assign_34^training/Adam/Assign_35^training/Adam/Assign_36^training/Adam/Assign_37^training/Adam/Assign_38^training/Adam/Assign_4^training/Adam/Assign_5^training/Adam/Assign_6^training/Adam/Assign_7^training/Adam/Assign_8^training/Adam/Assign_9

IsVariableInitializedIsVariableInitializedconv1d_1/kernel*"
_class
loc:@conv1d_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_1IsVariableInitializedconv1d_1/bias* 
_class
loc:@conv1d_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_2IsVariableInitializedconv1d_2/kernel*"
_class
loc:@conv1d_2/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_3IsVariableInitializedconv1d_2/bias* 
_class
loc:@conv1d_2/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_4IsVariableInitializedlstm_1/kernel* 
_class
loc:@lstm_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_5IsVariableInitializedlstm_1/recurrent_kernel**
_class 
loc:@lstm_1/recurrent_kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_6IsVariableInitializedlstm_1/bias*
_class
loc:@lstm_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_7IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_8IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_9IsVariableInitializeddense_2/kernel*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_10IsVariableInitializeddense_2/bias*
_class
loc:@dense_2/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_11IsVariableInitializeddense_3/kernel*
_output_shapes
: *!
_class
loc:@dense_3/kernel*
dtype0

IsVariableInitialized_12IsVariableInitializeddense_3/bias*
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_13IsVariableInitializedAdam/iterations*"
_class
loc:@Adam/iterations*
dtype0	*
_output_shapes
: 
{
IsVariableInitialized_14IsVariableInitializedAdam/lr*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 

IsVariableInitialized_15IsVariableInitializedAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_16IsVariableInitializedAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 

IsVariableInitialized_17IsVariableInitialized
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 

IsVariableInitialized_18IsVariableInitializedtraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
dtype0*
_output_shapes
: 

IsVariableInitialized_19IsVariableInitializedtraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_20IsVariableInitializedtraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0*
_output_shapes
: 

IsVariableInitialized_21IsVariableInitializedtraining/Adam/Variable_3*+
_class!
loc:@training/Adam/Variable_3*
dtype0*
_output_shapes
: 

IsVariableInitialized_22IsVariableInitializedtraining/Adam/Variable_4*+
_class!
loc:@training/Adam/Variable_4*
dtype0*
_output_shapes
: 

IsVariableInitialized_23IsVariableInitializedtraining/Adam/Variable_5*+
_class!
loc:@training/Adam/Variable_5*
dtype0*
_output_shapes
: 

IsVariableInitialized_24IsVariableInitializedtraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*
dtype0*
_output_shapes
: 

IsVariableInitialized_25IsVariableInitializedtraining/Adam/Variable_7*+
_class!
loc:@training/Adam/Variable_7*
dtype0*
_output_shapes
: 

IsVariableInitialized_26IsVariableInitializedtraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
dtype0*
_output_shapes
: 

IsVariab