       ЃK"	  РЭћжAbrain.Event:2(]Hѓм     ћ;Щ	агѕЭћжA"цЙ
y
lstm_1_inputPlaceholder*,
_output_shapes
:џџџџџџџџџ*!
shape:џџџџџџџџџ*
dtype0
l
lstm_1/random_uniform/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
^
lstm_1/random_uniform/minConst*
valueB
 *JQкО*
dtype0*
_output_shapes
: 
^
lstm_1/random_uniform/maxConst*
valueB
 *JQк>*
dtype0*
_output_shapes
: 
І
#lstm_1/random_uniform/RandomUniformRandomUniformlstm_1/random_uniform/shape*
T0*
dtype0*
_output_shapes

: *
seed2ќ*
seedБџх)
w
lstm_1/random_uniform/subSublstm_1/random_uniform/maxlstm_1/random_uniform/min*
_output_shapes
: *
T0

lstm_1/random_uniform/mulMul#lstm_1/random_uniform/RandomUniformlstm_1/random_uniform/sub*
T0*
_output_shapes

: 
{
lstm_1/random_uniformAddlstm_1/random_uniform/mullstm_1/random_uniform/min*
T0*
_output_shapes

: 

lstm_1/kernel
VariableV2*
_output_shapes

: *
	container *
shape
: *
shared_name *
dtype0
И
lstm_1/kernel/AssignAssignlstm_1/kernellstm_1/random_uniform* 
_class
loc:@lstm_1/kernel*
validate_shape(*
_output_shapes

: *
use_locking(*
T0
x
lstm_1/kernel/readIdentitylstm_1/kernel*
_output_shapes

: *
T0* 
_class
loc:@lstm_1/kernel
њ
%lstm_1/recurrent_kernel/initial_valueConst*
dtype0*
_output_shapes

: *
valueB "№Gб=qЯ	Н$MЂММ7>ЄxоННІНC=,>f#шН3y=^jzН_ОЪ5>SRЉ=-яIОкЗ>Пa6ОГк=іНРvОб@НYв=Hс?<Eх>HВОNК>
­=кuОљ>N3M>"\Y>І/Н <OмМНЫчИН!JDНPсЪ>жK=^ЛїЏНкWНќ%+ОQ>M>м2 =ЉMЉНZ>Г#О§њ$ОUдМQ>иW= а>ёE>=ОО}M>JсМ'ННr>р[ыН c9<чr>пъООе-Д=№>YЙ=К!ОфICОИкМmОGПНИОћНъ)mН@7 МUЧН/-GИLЄ=ђyОЪNЋ<u>ОЮУЇ> LОHНЁМ_ИxО[3НѕuОiБМНW=&EF>щRЩ=Пyт=l&ѓ>Т=Ўo>є;>уйНЛ:>TuЌНПср>>+(	>Щ>^Нci(ОиMSМ~їYОљa=ОКсN>cЎЧМcО}ЋОtR>Г=@џ$О^дЬН}ОкФОхaIО\ъЊНCLОЄЬНSWО-<>ЌОX{ЙМM==РкђНёn=+WМ	Њ6ОTОo'ОыОAТНYJ=лу!<б8>&МX[>)YМхї}>)оО1>у1>ўЦг=чgМuЇ=>8$Оb2О,тщ=	ЂЎ=йа:=лxA<#>IЯб>ЦDОВU=ї n;у.ОИиЇМp@Оy=ёџ<ъНиЉОТYFНш(Є>щ:НБ1Ч=ГеЊМѕC>.мъ= Ў<yКОOОФ(=О_b-ОњЮЖМuD:Н&Cг=хџA>§!ОЩЊ=ОїэО,еОАWЛ0ЅОјЇКђЯcО>$=CЕ=P>ЙF=ъ<5> ЃО	рЭ>">ЭjНbФтНШмНЊНj/ОсЯ\>Ъ>=њYОWэпНѕЙ<О9И>k*=]'Нlр=CНХЧНБv7Н эуМфЯ+>pшu>pо=>лZП>[і= ЛВНдО1>	Оy­=|=>жЖ<[ИНbн
>В
=AУ;О/жНљР= L=>В,<МTОЫ<мОЛњ)>&ЌН$Ор ОЏ6>8zНNН4>ыЊEНВДK>F}ОЛНкQОoСLО

lstm_1/recurrent_kernel
VariableV2*
shared_name *
dtype0*
_output_shapes

: *
	container *
shape
: 
ц
lstm_1/recurrent_kernel/AssignAssignlstm_1/recurrent_kernel%lstm_1/recurrent_kernel/initial_value*
use_locking(*
T0**
_class 
loc:@lstm_1/recurrent_kernel*
validate_shape(*
_output_shapes

: 

lstm_1/recurrent_kernel/readIdentitylstm_1/recurrent_kernel*
_output_shapes

: *
T0**
_class 
loc:@lstm_1/recurrent_kernel
Y
lstm_1/ConstConst*
dtype0*
_output_shapes
:*
valueB*    
[
lstm_1/Const_1Const*
valueB*  ?*
dtype0*
_output_shapes
:
[
lstm_1/Const_2Const*
_output_shapes
:*
valueB*    *
dtype0
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
: *

Tidx0*
T0
w
lstm_1/bias
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
І
lstm_1/bias/AssignAssignlstm_1/biaslstm_1/concat*
use_locking(*
T0*
_class
loc:@lstm_1/bias*
validate_shape(*
_output_shapes
: 
n
lstm_1/bias/readIdentitylstm_1/bias*
_output_shapes
: *
T0*
_class
loc:@lstm_1/bias
k
lstm_1/strided_slice/stackConst*
_output_shapes
:*
valueB"        *
dtype0
m
lstm_1/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
m
lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
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

:
m
lstm_1/strided_slice_1/stackConst*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_1/stack_1Const*
valueB"       *
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
lstm_1/strided_slice_1StridedSlicelstm_1/kernel/readlstm_1/strided_slice_1/stacklstm_1/strided_slice_1/stack_1lstm_1/strided_slice_1/stack_2*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:*
T0*
Index0*
shrink_axis_mask 
m
lstm_1/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB"       
o
lstm_1/strided_slice_2/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
В
lstm_1/strided_slice_2StridedSlicelstm_1/kernel/readlstm_1/strided_slice_2/stacklstm_1/strided_slice_2/stack_1lstm_1/strided_slice_2/stack_2*
_output_shapes

:*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
m
lstm_1/strided_slice_3/stackConst*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        
o
lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
В
lstm_1/strided_slice_3StridedSlicelstm_1/kernel/readlstm_1/strided_slice_3/stacklstm_1/strided_slice_3/stack_1lstm_1/strided_slice_3/stack_2*
_output_shapes

:*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
m
lstm_1/strided_slice_4/stackConst*
dtype0*
_output_shapes
:*
valueB"        
o
lstm_1/strided_slice_4/stack_1Const*
valueB"       *
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
lstm_1/strided_slice_4StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_4/stacklstm_1/strided_slice_4/stack_1lstm_1/strided_slice_4/stack_2*
end_mask*
_output_shapes

:*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask 
m
lstm_1/strided_slice_5/stackConst*
_output_shapes
:*
valueB"       *
dtype0
o
lstm_1/strided_slice_5/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
М
lstm_1/strided_slice_5StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_5/stacklstm_1/strided_slice_5/stack_1lstm_1/strided_slice_5/stack_2*
_output_shapes

:*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
m
lstm_1/strided_slice_6/stackConst*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_6/stack_1Const*
valueB"       *
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
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:*
Index0*
T0
m
lstm_1/strided_slice_7/stackConst*
valueB"       *
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
lstm_1/strided_slice_7/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
М
lstm_1/strided_slice_7StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_7/stacklstm_1/strided_slice_7/stack_1lstm_1/strided_slice_7/stack_2*
new_axis_mask *
end_mask*
_output_shapes

:*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask
f
lstm_1/strided_slice_8/stackConst*
valueB: *
dtype0*
_output_shapes
:
h
lstm_1/strided_slice_8/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
h
lstm_1/strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ќ
lstm_1/strided_slice_8StridedSlicelstm_1/bias/readlstm_1/strided_slice_8/stacklstm_1/strided_slice_8/stack_1lstm_1/strided_slice_8/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0
f
lstm_1/strided_slice_9/stackConst*
valueB:*
dtype0*
_output_shapes
:
h
lstm_1/strided_slice_9/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
h
lstm_1/strided_slice_9/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ќ
lstm_1/strided_slice_9StridedSlicelstm_1/bias/readlstm_1/strided_slice_9/stacklstm_1/strided_slice_9/stack_1lstm_1/strided_slice_9/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
:
g
lstm_1/strided_slice_10/stackConst*
_output_shapes
:*
valueB:*
dtype0
i
lstm_1/strided_slice_10/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
i
lstm_1/strided_slice_10/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
А
lstm_1/strided_slice_10StridedSlicelstm_1/bias/readlstm_1/strided_slice_10/stacklstm_1/strided_slice_10/stack_1lstm_1/strided_slice_10/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0*
shrink_axis_mask 
g
lstm_1/strided_slice_11/stackConst*
valueB:*
dtype0*
_output_shapes
:
i
lstm_1/strided_slice_11/stack_1Const*
_output_shapes
:*
valueB: *
dtype0
i
lstm_1/strided_slice_11/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
А
lstm_1/strided_slice_11StridedSlicelstm_1/bias/readlstm_1/strided_slice_11/stacklstm_1/strided_slice_11/stack_1lstm_1/strided_slice_11/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes
:
c
lstm_1/zeros_like	ZerosLikelstm_1_input*
T0*,
_output_shapes
:џџџџџџџџџ
m
lstm_1/Sum/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:
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
lstm_1/Sumlstm_1/ExpandDims/dim*'
_output_shapes
:џџџџџџџџџ*

Tdim0*
T0
f
lstm_1/Tile/multiplesConst*
dtype0*
_output_shapes
:*
valueB"      

lstm_1/TileTilelstm_1/ExpandDimslstm_1/Tile/multiples*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
h
lstm_1/Tile_1/multiplesConst*
valueB"      *
dtype0*
_output_shapes
:

lstm_1/Tile_1Tilelstm_1/ExpandDimslstm_1/Tile_1/multiples*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
j
lstm_1/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:

lstm_1/transpose	Transposelstm_1_inputlstm_1/transpose/perm*,
_output_shapes
:џџџџџџџџџ*
Tperm0*
T0
\
lstm_1/ShapeShapelstm_1/transpose*
T0*
out_type0*
_output_shapes
:
g
lstm_1/strided_slice_12/stackConst*
valueB: *
dtype0*
_output_shapes
:
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
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
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
lstm_1/strided_slice_13StridedSlicelstm_1/transposelstm_1/strided_slice_13/stacklstm_1/strided_slice_13/stack_1lstm_1/strided_slice_13/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMulMatMullstm_1/strided_slice_13lstm_1/strided_slice*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ђ
lstm_1/MatMul_1MatMullstm_1/strided_slice_13lstm_1/strided_slice_1*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ђ
lstm_1/MatMul_2MatMullstm_1/strided_slice_13lstm_1/strided_slice_2*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
Ђ
lstm_1/MatMul_3MatMullstm_1/strided_slice_13lstm_1/strided_slice_3*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 

lstm_1/BiasAddBiasAddlstm_1/MatMullstm_1/strided_slice_8*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ

lstm_1/BiasAdd_1BiasAddlstm_1/MatMul_1lstm_1/strided_slice_9*'
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC

lstm_1/BiasAdd_2BiasAddlstm_1/MatMul_2lstm_1/strided_slice_10*'
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC

lstm_1/BiasAdd_3BiasAddlstm_1/MatMul_3lstm_1/strided_slice_11*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0

lstm_1/MatMul_4MatMullstm_1/Tilelstm_1/strided_slice_4*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
d

lstm_1/addAddlstm_1/BiasAddlstm_1/MatMul_4*
T0*'
_output_shapes
:џџџџџџџџџ
Q
lstm_1/mul/xConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
]

lstm_1/mulMullstm_1/mul/x
lstm_1/add*'
_output_shapes
:џџџџџџџџџ*
T0
S
lstm_1/add_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *   ?
a
lstm_1/add_1Add
lstm_1/mullstm_1/add_1/y*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/Const_3Const*
valueB
 *    *
dtype0*
_output_shapes
: 
S
lstm_1/Const_4Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
w
lstm_1/clip_by_value/MinimumMinimumlstm_1/add_1lstm_1/Const_4*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/clip_by_valueMaximumlstm_1/clip_by_value/Minimumlstm_1/Const_3*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMul_5MatMullstm_1/Tilelstm_1/strided_slice_5*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
h
lstm_1/add_2Addlstm_1/BiasAdd_1lstm_1/MatMul_5*'
_output_shapes
:џџџџџџџџџ*
T0
S
lstm_1/mul_1/xConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
c
lstm_1/mul_1Mullstm_1/mul_1/xlstm_1/add_2*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/add_3/yConst*
dtype0*
_output_shapes
: *
valueB
 *   ?
c
lstm_1/add_3Addlstm_1/mul_1lstm_1/add_3/y*'
_output_shapes
:џџџџџџџџџ*
T0
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
lstm_1/clip_by_value_1/MinimumMinimumlstm_1/add_3lstm_1/Const_6*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/clip_by_value_1Maximumlstm_1/clip_by_value_1/Minimumlstm_1/Const_5*
T0*'
_output_shapes
:џџџџџџџџџ
l
lstm_1/mul_2Mullstm_1/clip_by_value_1lstm_1/Tile_1*'
_output_shapes
:џџџџџџџџџ*
T0

lstm_1/MatMul_6MatMullstm_1/Tilelstm_1/strided_slice_6*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
h
lstm_1/add_4Addlstm_1/BiasAdd_2lstm_1/MatMul_6*'
_output_shapes
:џџџџџџџџџ*
T0
S
lstm_1/TanhTanhlstm_1/add_4*'
_output_shapes
:џџџџџџџџџ*
T0
h
lstm_1/mul_3Mullstm_1/clip_by_valuelstm_1/Tanh*
T0*'
_output_shapes
:џџџџџџџџџ
a
lstm_1/add_5Addlstm_1/mul_2lstm_1/mul_3*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMul_7MatMullstm_1/Tilelstm_1/strided_slice_7*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
h
lstm_1/add_6Addlstm_1/BiasAdd_3lstm_1/MatMul_7*'
_output_shapes
:џџџџџџџџџ*
T0
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
:џџџџџџџџџ
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
:џџџџџџџџџ*
T0
S
lstm_1/Const_7Const*
valueB
 *    *
dtype0*
_output_shapes
: 
S
lstm_1/Const_8Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
y
lstm_1/clip_by_value_2/MinimumMinimumlstm_1/add_7lstm_1/Const_8*'
_output_shapes
:џџџџџџџџџ*
T0

lstm_1/clip_by_value_2Maximumlstm_1/clip_by_value_2/Minimumlstm_1/Const_7*
T0*'
_output_shapes
:џџџџџџџџџ
U
lstm_1/Tanh_1Tanhlstm_1/add_5*'
_output_shapes
:џџџџџџџџџ*
T0
l
lstm_1/mul_5Mullstm_1/clip_by_value_2lstm_1/Tanh_1*
T0*'
_output_shapes
:џџџџџџџџџ
ь
lstm_1/TensorArrayTensorArrayV3lstm_1/strided_slice_12* 
tensor_array_name	output_ta*
dtype0*
_output_shapes

:: *
element_shape:*
clear_after_read(*
dynamic_size( *
identical_element_shapes(
э
lstm_1/TensorArray_1TensorArrayV3lstm_1/strided_slice_12*
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*
tensor_array_name
input_ta*
dtype0*
_output_shapes

:: 
o
lstm_1/TensorArrayUnstack/ShapeShapelstm_1/transpose*
_output_shapes
:*
T0*
out_type0
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
'lstm_1/TensorArrayUnstack/strided_sliceStridedSlicelstm_1/TensorArrayUnstack/Shape-lstm_1/TensorArrayUnstack/strided_slice/stack/lstm_1/TensorArrayUnstack/strided_slice/stack_1/lstm_1/TensorArrayUnstack/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
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
lstm_1/TensorArrayUnstack/rangeRange%lstm_1/TensorArrayUnstack/range/start'lstm_1/TensorArrayUnstack/strided_slice%lstm_1/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:џџџџџџџџџ
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
lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
value
B :*
dtype0
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
lstm_1/while/Enter_2Enterlstm_1/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: **

frame_namelstm_1/while/while_context
Ж
lstm_1/while/Enter_3Enterlstm_1/Tile*
T0*
is_constant( *
parallel_iterations *'
_output_shapes
:џџџџџџџџџ**

frame_namelstm_1/while/while_context
И
lstm_1/while/Enter_4Enterlstm_1/Tile_1*
T0*
is_constant( *
parallel_iterations *'
_output_shapes
:џџџџџџџџџ**

frame_namelstm_1/while/while_context
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
T0*
N*)
_output_shapes
:џџџџџџџџџ: 

lstm_1/while/Merge_4Mergelstm_1/while/Enter_4lstm_1/while/NextIteration_4*
T0*
N*)
_output_shapes
:џџџџџџџџџ: 
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
lstm_1/while/Switch_3Switchlstm_1/while/Merge_3lstm_1/while/LoopCond*'
_class
loc:@lstm_1/while/Merge_3*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*
T0
К
lstm_1/while/Switch_4Switchlstm_1/while/Merge_4lstm_1/while/LoopCond*
T0*'
_class
loc:@lstm_1/while/Merge_4*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
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
lstm_1/while/Identity_3Identitylstm_1/while/Switch_3:1*
T0*'
_output_shapes
:џџџџџџџџџ
n
lstm_1/while/Identity_4Identitylstm_1/while/Switch_4:1*
T0*'
_output_shapes
:џџџџџџџџџ
l
lstm_1/while/add/yConst^lstm_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
c
lstm_1/while/addAddlstm_1/while/Identitylstm_1/while/add/y*
T0*
_output_shapes
: 
а
lstm_1/while/TensorArrayReadV3TensorArrayReadV3$lstm_1/while/TensorArrayReadV3/Enterlstm_1/while/Identity_1&lstm_1/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:џџџџџџџџџ
Т
$lstm_1/while/TensorArrayReadV3/EnterEnterlstm_1/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelstm_1/while/while_context
э
&lstm_1/while/TensorArrayReadV3/Enter_1EnterAlstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *
_output_shapes
: **

frame_namelstm_1/while/while_context*
T0*
is_constant(
А
lstm_1/while/MatMulMatMullstm_1/while/TensorArrayReadV3lstm_1/while/MatMul/Enter*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
Л
lstm_1/while/MatMul/EnterEnterlstm_1/strided_slice*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context*
T0*
is_constant(
Д
lstm_1/while/MatMul_1MatMullstm_1/while/TensorArrayReadV3lstm_1/while/MatMul_1/Enter*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
П
lstm_1/while/MatMul_1/EnterEnterlstm_1/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context
Д
lstm_1/while/MatMul_2MatMullstm_1/while/TensorArrayReadV3lstm_1/while/MatMul_2/Enter*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 
П
lstm_1/while/MatMul_2/EnterEnterlstm_1/strided_slice_2*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context
Д
lstm_1/while/MatMul_3MatMullstm_1/while/TensorArrayReadV3lstm_1/while/MatMul_3/Enter*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
П
lstm_1/while/MatMul_3/EnterEnterlstm_1/strided_slice_3*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context*
T0*
is_constant(

lstm_1/while/BiasAddBiasAddlstm_1/while/MatMullstm_1/while/BiasAdd/Enter*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
К
lstm_1/while/BiasAdd/EnterEnterlstm_1/strided_slice_8*
parallel_iterations *
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*
is_constant(

lstm_1/while/BiasAdd_1BiasAddlstm_1/while/MatMul_1lstm_1/while/BiasAdd_1/Enter*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
М
lstm_1/while/BiasAdd_1/EnterEnterlstm_1/strided_slice_9*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*
is_constant(*
parallel_iterations 

lstm_1/while/BiasAdd_2BiasAddlstm_1/while/MatMul_2lstm_1/while/BiasAdd_2/Enter*'
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
Н
lstm_1/while/BiasAdd_2/EnterEnterlstm_1/strided_slice_10*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0

lstm_1/while/BiasAdd_3BiasAddlstm_1/while/MatMul_3lstm_1/while/BiasAdd_3/Enter*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
Н
lstm_1/while/BiasAdd_3/EnterEnterlstm_1/strided_slice_11*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelstm_1/while/while_context
­
lstm_1/while/MatMul_4MatMullstm_1/while/Identity_3lstm_1/while/MatMul_4/Enter*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
П
lstm_1/while/MatMul_4/EnterEnterlstm_1/strided_slice_4*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context
x
lstm_1/while/add_1Addlstm_1/while/BiasAddlstm_1/while/MatMul_4*
T0*'
_output_shapes
:џџџџџџџџџ
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
:џџџџџџџџџ
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
:џџџџџџџџџ
o
lstm_1/while/ConstConst^lstm_1/while/Identity*
_output_shapes
: *
valueB
 *    *
dtype0
q
lstm_1/while/Const_1Const^lstm_1/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 

"lstm_1/while/clip_by_value/MinimumMinimumlstm_1/while/add_2lstm_1/while/Const_1*'
_output_shapes
:џџџџџџџџџ*
T0

lstm_1/while/clip_by_valueMaximum"lstm_1/while/clip_by_value/Minimumlstm_1/while/Const*'
_output_shapes
:џџџџџџџџџ*
T0
­
lstm_1/while/MatMul_5MatMullstm_1/while/Identity_3lstm_1/while/MatMul_5/Enter*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 
П
lstm_1/while/MatMul_5/EnterEnterlstm_1/strided_slice_5*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context*
T0*
is_constant(
z
lstm_1/while/add_3Addlstm_1/while/BiasAdd_1lstm_1/while/MatMul_5*'
_output_shapes
:џџџџџџџџџ*
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
:џџџџџџџџџ
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
:џџџџџџџџџ
q
lstm_1/while/Const_2Const^lstm_1/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *    
q
lstm_1/while/Const_3Const^lstm_1/while/Identity*
_output_shapes
: *
valueB
 *  ?*
dtype0

$lstm_1/while/clip_by_value_1/MinimumMinimumlstm_1/while/add_4lstm_1/while/Const_3*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/while/clip_by_value_1Maximum$lstm_1/while/clip_by_value_1/Minimumlstm_1/while/Const_2*'
_output_shapes
:џџџџџџџџџ*
T0

lstm_1/while/mul_2Mullstm_1/while/clip_by_value_1lstm_1/while/Identity_4*
T0*'
_output_shapes
:џџџџџџџџџ
­
lstm_1/while/MatMul_6MatMullstm_1/while/Identity_3lstm_1/while/MatMul_6/Enter*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
П
lstm_1/while/MatMul_6/EnterEnterlstm_1/strided_slice_6*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context
z
lstm_1/while/add_5Addlstm_1/while/BiasAdd_2lstm_1/while/MatMul_6*'
_output_shapes
:џџџџџџџџџ*
T0
_
lstm_1/while/TanhTanhlstm_1/while/add_5*
T0*'
_output_shapes
:џџџџџџџџџ
z
lstm_1/while/mul_3Mullstm_1/while/clip_by_valuelstm_1/while/Tanh*'
_output_shapes
:џџџџџџџџџ*
T0
s
lstm_1/while/add_6Addlstm_1/while/mul_2lstm_1/while/mul_3*'
_output_shapes
:џџџџџџџџџ*
T0
­
lstm_1/while/MatMul_7MatMullstm_1/while/Identity_3lstm_1/while/MatMul_7/Enter*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
П
lstm_1/while/MatMul_7/EnterEnterlstm_1/strided_slice_7*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context
z
lstm_1/while/add_7Addlstm_1/while/BiasAdd_3lstm_1/while/MatMul_7*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/mul_4/xConst^lstm_1/while/Identity*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
u
lstm_1/while/mul_4Mullstm_1/while/mul_4/xlstm_1/while/add_7*
T0*'
_output_shapes
:џџџџџџџџџ
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
:џџџџџџџџџ*
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
:џџџџџџџџџ

lstm_1/while/clip_by_value_2Maximum$lstm_1/while/clip_by_value_2/Minimumlstm_1/while/Const_4*
T0*'
_output_shapes
:џџџџџџџџџ
a
lstm_1/while/Tanh_1Tanhlstm_1/while/add_6*
T0*'
_output_shapes
:џџџџџџџџџ
~
lstm_1/while/mul_5Mullstm_1/while/clip_by_value_2lstm_1/while/Tanh_1*
T0*'
_output_shapes
:џџџџџџџџџ

0lstm_1/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV36lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/Enterlstm_1/while/Identity_1lstm_1/while/mul_5lstm_1/while/Identity_2*
T0*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
: 
љ
6lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterlstm_1/TensorArray*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5
n
lstm_1/while/add_9/yConst^lstm_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
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
lstm_1/while/NextIteration_3NextIterationlstm_1/while/mul_5*
T0*'
_output_shapes
:џџџџџџџџџ
s
lstm_1/while/NextIteration_4NextIterationlstm_1/while/add_6*'
_output_shapes
:џџџџџџџџџ*
T0
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
lstm_1/while/Exit_2Exitlstm_1/while/Switch_2*
_output_shapes
: *
T0
d
lstm_1/while/Exit_3Exitlstm_1/while/Switch_3*'
_output_shapes
:џџџџџџџџџ*
T0
d
lstm_1/while/Exit_4Exitlstm_1/while/Switch_4*'
_output_shapes
:џџџџџџџџџ*
T0
І
)lstm_1/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3lstm_1/TensorArraylstm_1/while/Exit_2*
_output_shapes
: *%
_class
loc:@lstm_1/TensorArray

#lstm_1/TensorArrayStack/range/startConst*%
_class
loc:@lstm_1/TensorArray*
value	B : *
dtype0*
_output_shapes
: 

#lstm_1/TensorArrayStack/range/deltaConst*
dtype0*
_output_shapes
: *%
_class
loc:@lstm_1/TensorArray*
value	B :
ѓ
lstm_1/TensorArrayStack/rangeRange#lstm_1/TensorArrayStack/range/start)lstm_1/TensorArrayStack/TensorArraySizeV3#lstm_1/TensorArrayStack/range/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0*%
_class
loc:@lstm_1/TensorArray

+lstm_1/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3lstm_1/TensorArraylstm_1/TensorArrayStack/rangelstm_1/while/Exit_2*%
_class
loc:@lstm_1/TensorArray*
dtype0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*$
element_shape:џџџџџџџџџ
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
:џџџџџџџџџ
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
 :џџџџџџџџџџџџџџџџџџ*
Tperm0
a
flatten_1/ShapeShapelstm_1/transpose_1*
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
flatten_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Џ
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes
:
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
flatten_1/stack/0Const*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
t
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*
N*
_output_shapes
:*
T0*

axis 

flatten_1/ReshapeReshapelstm_1/transpose_1flatten_1/stack*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
m
dense_1/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
_
dense_1/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *:ЭН
_
dense_1/random_uniform/maxConst*
valueB
 *:Э=*
dtype0*
_output_shapes
: 
Њ
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
T0*
dtype0* 
_output_shapes
:
*
seed2ЎТц*
seedБџх)
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
_output_shapes
: *
T0

dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0* 
_output_shapes
:


dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min* 
_output_shapes
:
*
T0

dense_1/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
	container *
shape:
*
shared_name 
О
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*!
_class
loc:@dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
}
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel* 
_output_shapes
:

\
dense_1/ConstConst*
_output_shapes	
:*
valueB*    *
dtype0
z
dense_1/bias
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
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
dense_1/MatMulMatMulflatten_1/Reshapedense_1/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*(
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
X
dense_1/ReluReludense_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
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
dropout_1/cond/switch_tIdentitydropout_1/cond/Switch:1*
T0
*
_output_shapes
: 
[
dropout_1/cond/switch_fIdentitydropout_1/cond/Switch*
_output_shapes
: *
T0

c
dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
: *
T0

s
dropout_1/cond/mul/yConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?

dropout_1/cond/mulMuldropout_1/cond/mul/Switch:1dropout_1/cond/mul/y*
T0*(
_output_shapes
:џџџџџџџџџ
Б
dropout_1/cond/mul/SwitchSwitchdense_1/Reludropout_1/cond/pred_id*
T0*
_class
loc:@dense_1/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ

 dropout_1/cond/dropout/keep_probConst^dropout_1/cond/switch_t*
_output_shapes
: *
valueB
 *   ?*
dtype0
n
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
T0*
out_type0*
_output_shapes
:
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
С
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
dtype0*(
_output_shapes
:џџџџџџџџџ*
seed2ќЫњ*
seedБџх)*
T0
Ї
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
У
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*(
_output_shapes
:џџџџџџџџџ*
T0
Е
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_1/cond/dropout/addAdd dropout_1/cond/dropout/keep_prob%dropout_1/cond/dropout/random_uniform*
T0*(
_output_shapes
:џџџџџџџџџ
t
dropout_1/cond/dropout/FloorFloordropout_1/cond/dropout/add*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_1/cond/dropout/divRealDivdropout_1/cond/mul dropout_1/cond/dropout/keep_prob*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/divdropout_1/cond/dropout/Floor*
T0*(
_output_shapes
:џџџџџџџџџ
Џ
dropout_1/cond/Switch_1Switchdense_1/Reludropout_1/cond/pred_id*
T0*
_class
loc:@dense_1/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ

dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*
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
dense_2/random_uniform/minConst*
valueB
 *qФО*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
valueB
 *qФ>*
dtype0*
_output_shapes
: 
Њ
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
T0*
dtype0* 
_output_shapes
:
*
seed2кзШ*
seedБџх)
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
VariableV2* 
_output_shapes
:
*
	container *
shape:
*
shared_name *
dtype0
О
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*!
_class
loc:@dense_2/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
}
dense_2/kernel/readIdentitydense_2/kernel* 
_output_shapes
:
*
T0*!
_class
loc:@dense_2/kernel
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
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes	
:

dense_2/MatMulMatMuldropout_1/cond/Mergedense_2/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ*
T0
X
dense_2/ReluReludense_2/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0

dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes
: : 
]
dropout_2/cond/switch_tIdentitydropout_2/cond/Switch:1*
_output_shapes
: *
T0

[
dropout_2/cond/switch_fIdentitydropout_2/cond/Switch*
_output_shapes
: *
T0

c
dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
: 
s
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?

dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*
T0*(
_output_shapes
:џџџџџџџџџ
Б
dropout_2/cond/mul/SwitchSwitchdense_2/Reludropout_2/cond/pred_id*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*
T0*
_class
loc:@dense_2/Relu

 dropout_2/cond/dropout/keep_probConst^dropout_2/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
n
dropout_2/cond/dropout/ShapeShapedropout_2/cond/mul*
out_type0*
_output_shapes
:*
T0

)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    

)dropout_2/cond/dropout/random_uniform/maxConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
С
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
dtype0*(
_output_shapes
:џџџџџџџџџ*
seed2ъУЋ*
seedБџх)*
T0
Ї
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
У
)dropout_2/cond/dropout/random_uniform/mulMul3dropout_2/cond/dropout/random_uniform/RandomUniform)dropout_2/cond/dropout/random_uniform/sub*(
_output_shapes
:џџџџџџџџџ*
T0
Е
%dropout_2/cond/dropout/random_uniformAdd)dropout_2/cond/dropout/random_uniform/mul)dropout_2/cond/dropout/random_uniform/min*(
_output_shapes
:џџџџџџџџџ*
T0

dropout_2/cond/dropout/addAdd dropout_2/cond/dropout/keep_prob%dropout_2/cond/dropout/random_uniform*
T0*(
_output_shapes
:џџџџџџџџџ
t
dropout_2/cond/dropout/FloorFloordropout_2/cond/dropout/add*
T0*(
_output_shapes
:џџџџџџџџџ
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
dropout_2/cond/Switch_1Switchdense_2/Reludropout_2/cond/pred_id*
_class
loc:@dense_2/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*
T0

dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul*
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
dense_3/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *ЃЎXО
_
dense_3/random_uniform/maxConst*
valueB
 *ЃЎX>*
dtype0*
_output_shapes
: 
Љ
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
dtype0*
_output_shapes
:	*
seed2Д*
seedБџх)*
T0
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
_output_shapes
: *
T0
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
VariableV2*
shape:	*
shared_name *
dtype0*
_output_shapes
:	*
	container 
Н
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes
:	
|
dense_3/kernel/readIdentitydense_3/kernel*!
_class
loc:@dense_3/kernel*
_output_shapes
:	*
T0
Z
dense_3/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_3/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
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
dense_3/MatMulMatMuldropout_2/cond/Mergedense_3/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
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
Adam/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
s
Adam/iterations
VariableV2*
dtype0	*
_output_shapes
: *
	container *
shape: *
shared_name 
О
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*"
_class
loc:@Adam/iterations*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	
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
Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/lr*
validate_shape(
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
Adam/beta_1/readIdentityAdam/beta_1*
_output_shapes
: *
T0*
_class
loc:@Adam/beta_1
^
Adam/beta_2/initial_valueConst*
valueB
 *wО?*
dtype0*
_output_shapes
: 
o
Adam/beta_2
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
Ў
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
_class
loc:@Adam/beta_2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
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
Adam/decay*
T0*
_class
loc:@Adam/decay*
_output_shapes
: 

dense_3_targetPlaceholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
q
dense_3_sample_weightsPlaceholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
r
'loss/dense_3_loss/Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ
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
loss/dense_3_loss/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *Пж3
\
loss/dense_3_loss/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
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
loss/dense_3_loss/Sum_1Sumloss/dense_3_loss/mul)loss/dense_3_loss/Sum_1/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( *

Tidx0
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
loss/dense_3_loss/Mean_1Meanloss/dense_3_loss/Castloss/dense_3_loss/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
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
loss/dense_3_loss/Mean_2Meanloss/dense_3_loss/truediv_1loss/dense_3_loss/Const_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
O

loss/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
V
loss/mulMul
loss/mul/xloss/dense_3_loss/Mean_2*
_output_shapes
: *
T0
g
metrics/acc/ArgMax/dimensionConst*
_output_shapes
: *
valueB :
џџџџџџџџџ*
dtype0

metrics/acc/ArgMaxArgMaxdense_3_targetmetrics/acc/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:џџџџџџџџџ*

Tidx0
i
metrics/acc/ArgMax_1/dimensionConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

metrics/acc/ArgMax_1ArgMaxdense_3/Softmaxmetrics/acc/ArgMax_1/dimension*
output_type0	*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
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
metrics/acc/ConstConst*
_output_shapes
:*
valueB: *
dtype0
{
metrics/acc/MeanMeanmetrics/acc/Castmetrics/acc/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
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
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
_output_shapes
: *
T0*
_class
loc:@loss/mul*

index_type0

training/Adam/gradients/f_countConst*&
_class
loc:@lstm_1/while/Exit_2*
value	B : *
dtype0*
_output_shapes
: 
ю
!training/Adam/gradients/f_count_1Entertraining/Adam/gradients/f_count*
_output_shapes
: **

frame_namelstm_1/while/while_context*
T0*&
_class
loc:@lstm_1/while/Exit_2*
parallel_iterations *
is_constant( 
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
training/Adam/gradients/AddAdd training/Adam/gradients/Switch:1training/Adam/gradients/Add/y*
_output_shapes
: *
T0*&
_class
loc:@lstm_1/while/Exit_2

%training/Adam/gradients/NextIterationNextIterationtraining/Adam/gradients/AddH^training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPushV2H^training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPushV2l^training/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2R^training/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPushV2_1R^training/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPushV2R^training/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPushV2_1R^training/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPushV2R^training/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPushV2_1R^training/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPushV2_1R^training/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPushV2_1R^training/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPushV2b^training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPushV2V^training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPushV2d^training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPushV2X^training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPushV2\^training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPushV2S^training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPushV2d^training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPushV2X^training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPushV2\^training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPushV2S^training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPushV2Z^training/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPushV2Q^training/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPushV2R^training/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPushV2@^training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPushV2R^training/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPushV2_1@^training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPushV2B^training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPushV2R^training/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPushV2_1@^training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPushV2B^training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPushV2R^training/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPushV2@^training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPushV2R^training/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPushV2_1@^training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPushV2B^training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPushV2P^training/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPushV2>^training/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPushV2*
T0*&
_class
loc:@lstm_1/while/Exit_2*
_output_shapes
: 

!training/Adam/gradients/f_count_2Exittraining/Adam/gradients/Switch*
T0*&
_class
loc:@lstm_1/while/Exit_2*
_output_shapes
: 

training/Adam/gradients/b_countConst*
_output_shapes
: *&
_class
loc:@lstm_1/while/Exit_2*
value	B :*
dtype0

!training/Adam/gradients/b_count_1Enter!training/Adam/gradients/f_count_2*
T0*&
_class
loc:@lstm_1/while/Exit_2*
parallel_iterations *
is_constant( *
_output_shapes
: *B

frame_name42training/Adam/gradients/lstm_1/while/while_context
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
training/Adam/gradients/SubSub"training/Adam/gradients/Switch_1:1*training/Adam/gradients/GreaterEqual/Enter*
T0*&
_class
loc:@lstm_1/while/Exit_2*
_output_shapes
: 

'training/Adam/gradients/NextIteration_1NextIterationtraining/Adam/gradients/Subg^training/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0*&
_class
loc:@lstm_1/while/Exit_2*
_output_shapes
: 

!training/Adam/gradients/b_count_3Exit training/Adam/gradients/Switch_1*
T0*&
_class
loc:@lstm_1/while/Exit_2*
_output_shapes
: 
І
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/dense_3_loss/Mean_2*
_class
loc:@loss/mul*
_output_shapes
: *
T0

+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
_class
loc:@loss/mul*
_output_shapes
: *
T0
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
:training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/TileTile=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Reshape;training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Shape*

Tmultiples0*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*#
_output_shapes
:џџџџџџџџџ
Х
=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Shape_1Shapeloss/dense_3_loss/truediv_1*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*
out_type0*
_output_shapes
:
­
=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Shape_2Const*+
_class!
loc:@loss/dense_3_loss/Mean_2*
valueB *
dtype0*
_output_shapes
: 
В
;training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/ConstConst*
_output_shapes
:*+
_class!
loc:@loss/dense_3_loss/Mean_2*
valueB: *
dtype0
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
<training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Prod_1Prod=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Shape_2=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Const_1*
	keep_dims( *

Tidx0*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*
_output_shapes
: 
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
>training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/ShapeShapeloss/dense_3_loss/mul_1*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
out_type0*
_output_shapes
:
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
@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDivRealDiv=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/truedivloss/dense_3_loss/Mean_1*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ
Х
<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/SumSum@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDivNtraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/BroadcastGradientArgs*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
_output_shapes
:*
	keep_dims( *

Tidx0
Е
@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/ReshapeReshape<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Sum>training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Shape*#
_output_shapes
:џџџџџџџџџ*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
Tshape0
К
<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/NegNegloss/dense_3_loss/mul_1*#
_output_shapes
:џџџџџџџџџ*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1

Btraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDiv_1RealDiv<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Negloss/dense_3_loss/Mean_1*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ

Btraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDiv_2RealDivBtraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDiv_1loss/dense_3_loss/Mean_1*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ
Є
<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/mulMul=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/truedivBtraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDiv_2*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ
Х
>training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Sum_1Sum<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/mulPtraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/BroadcastGradientArgs:1*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
_output_shapes
:*
	keep_dims( *

Tidx0
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
<training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Shape_1Shapedense_3_sample_weights*
_output_shapes
:*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*
out_type0
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
8training/Adam/gradients/loss/dense_3_loss/mul_1_grad/SumSum8training/Adam/gradients/loss/dense_3_loss/mul_1_grad/MulJtraining/Adam/gradients/loss/dense_3_loss/mul_1_grad/BroadcastGradientArgs**
_class 
loc:@loss/dense_3_loss/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ѕ
<training/Adam/gradients/loss/dense_3_loss/mul_1_grad/ReshapeReshape8training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Sum:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Shape*#
_output_shapes
:џџџџџџџџџ*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*
Tshape0
ѕ
:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Mul_1Mulloss/dense_3_loss/Mean@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Reshape*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*#
_output_shapes
:џџџџџџџџџ
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
>training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Reshape_1Reshape:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Sum_1<training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Shape_1*#
_output_shapes
:џџџџџџџџџ*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*
Tshape0
Й
9training/Adam/gradients/loss/dense_3_loss/Mean_grad/ShapeShapeloss/dense_3_loss/Neg*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
out_type0*
_output_shapes
:
Ѕ
8training/Adam/gradients/loss/dense_3_loss/Mean_grad/SizeConst*
dtype0*
_output_shapes
: *)
_class
loc:@loss/dense_3_loss/Mean*
value	B :
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
8training/Adam/gradients/loss/dense_3_loss/Mean_grad/FillFill;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_1>training/Adam/gradients/loss/dense_3_loss/Mean_grad/Fill/value*
_output_shapes
: *
T0*)
_class
loc:@loss/dense_3_loss/Mean*

index_type0
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
<training/Adam/gradients/loss/dense_3_loss/Mean_grad/floordivFloorDiv9training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum*
_output_shapes
:*
T0*)
_class
loc:@loss/dense_3_loss/Mean
Ў
;training/Adam/gradients/loss/dense_3_loss/Mean_grad/ReshapeReshape<training/Adam/gradients/loss/dense_3_loss/mul_1_grad/ReshapeAtraining/Adam/gradients/loss/dense_3_loss/Mean_grad/DynamicStitch*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
Tshape0*#
_output_shapes
:џџџџџџџџџ
І
8training/Adam/gradients/loss/dense_3_loss/Mean_grad/TileTile;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Reshape<training/Adam/gradients/loss/dense_3_loss/Mean_grad/floordiv*)
_class
loc:@loss/dense_3_loss/Mean*#
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
Л
;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_2Shapeloss/dense_3_loss/Neg*)
_class
loc:@loss/dense_3_loss/Mean*
out_type0*
_output_shapes
:*
T0
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
8training/Adam/gradients/loss/dense_3_loss/Mean_grad/ProdProd;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_29training/Adam/gradients/loss/dense_3_loss/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*)
_class
loc:@loss/dense_3_loss/Mean
А
;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Const_1Const*
_output_shapes
:*)
_class
loc:@loss/dense_3_loss/Mean*
valueB: *
dtype0
Ѕ
:training/Adam/gradients/loss/dense_3_loss/Mean_grad/Prod_1Prod;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_3;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Const_1*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
Ќ
?training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum_1/yConst*)
_class
loc:@loss/dense_3_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 

=training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum_1Maximum:training/Adam/gradients/loss/dense_3_loss/Mean_grad/Prod_1?training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum_1/y*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 

>training/Adam/gradients/loss/dense_3_loss/Mean_grad/floordiv_1FloorDiv8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Prod=training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum_1*
_output_shapes
: *
T0*)
_class
loc:@loss/dense_3_loss/Mean
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
@training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/range/deltaConst*
dtype0*
_output_shapes
: **
_class 
loc:@loss/dense_3_loss/Sum_1*
value	B :
ж
:training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/rangeRange@training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/range/start9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Size@training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/range/delta*

Tidx0**
_class 
loc:@loss/dense_3_loss/Sum_1*
_output_shapes
:
­
?training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Fill/valueConst**
_class 
loc:@loss/dense_3_loss/Sum_1*
value	B :*
dtype0*
_output_shapes
: 

9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/FillFill<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Shape_1?training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Fill/value*
_output_shapes
: *
T0**
_class 
loc:@loss/dense_3_loss/Sum_1*

index_type0
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
=training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/floordivFloorDiv:training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Shape<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Maximum*
T0**
_class 
loc:@loss/dense_3_loss/Sum_1*
_output_shapes
:
И
<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/ReshapeReshape6training/Adam/gradients/loss/dense_3_loss/Neg_grad/NegBtraining/Adam/gradients/loss/dense_3_loss/Sum_1_grad/DynamicStitch**
_class 
loc:@loss/dense_3_loss/Sum_1*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
Ў
9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/TileTile<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Reshape=training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/floordiv**
_class 
loc:@loss/dense_3_loss/Sum_1*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
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
6training/Adam/gradients/loss/dense_3_loss/mul_grad/MulMul9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Tileloss/dense_3_loss/Log*(
_class
loc:@loss/dense_3_loss/mul*'
_output_shapes
:џџџџџџџџџ*
T0
Љ
6training/Adam/gradients/loss/dense_3_loss/mul_grad/SumSum6training/Adam/gradients/loss/dense_3_loss/mul_grad/MulHtraining/Adam/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*(
_class
loc:@loss/dense_3_loss/mul*
_output_shapes
:
Њ
:training/Adam/gradients/loss/dense_3_loss/mul_grad/ReshapeReshape6training/Adam/gradients/loss/dense_3_loss/mul_grad/Sum8training/Adam/gradients/loss/dense_3_loss/mul_grad/Shape*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0*(
_class
loc:@loss/dense_3_loss/mul*
Tshape0
ц
8training/Adam/gradients/loss/dense_3_loss/mul_grad/Mul_1Muldense_3_target9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Tile*
T0*(
_class
loc:@loss/dense_3_loss/mul*'
_output_shapes
:џџџџџџџџџ
Џ
8training/Adam/gradients/loss/dense_3_loss/mul_grad/Sum_1Sum8training/Adam/gradients/loss/dense_3_loss/mul_grad/Mul_1Jtraining/Adam/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*(
_class
loc:@loss/dense_3_loss/mul
Ї
<training/Adam/gradients/loss/dense_3_loss/mul_grad/Reshape_1Reshape8training/Adam/gradients/loss/dense_3_loss/mul_grad/Sum_1:training/Adam/gradients/loss/dense_3_loss/mul_grad/Shape_1*
T0*(
_class
loc:@loss/dense_3_loss/mul*
Tshape0*'
_output_shapes
:џџџџџџџџџ

=training/Adam/gradients/loss/dense_3_loss/Log_grad/Reciprocal
Reciprocalloss/dense_3_loss/clip_by_value=^training/Adam/gradients/loss/dense_3_loss/mul_grad/Reshape_1*(
_class
loc:@loss/dense_3_loss/Log*'
_output_shapes
:џџџџџџџџџ*
T0

6training/Adam/gradients/loss/dense_3_loss/Log_grad/mulMul<training/Adam/gradients/loss/dense_3_loss/mul_grad/Reshape_1=training/Adam/gradients/loss/dense_3_loss/Log_grad/Reciprocal*'
_output_shapes
:џџџџџџџџџ*
T0*(
_class
loc:@loss/dense_3_loss/Log
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
Htraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/zeros/ConstConst*
_output_shapes
: *2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
valueB
 *    *
dtype0
в
Btraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/zerosFillDtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape_2Htraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/zeros/Const*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*

index_type0*'
_output_shapes
:џџџџџџџџџ

Itraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/GreaterEqualGreaterEqual'loss/dense_3_loss/clip_by_value/Minimumloss/dense_3_loss/Const*'
_output_shapes
:џџџџџџџџџ*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value
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
Dtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/ReshapeReshape@training/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/SumBtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
Tshape0*'
_output_shapes
:џџџџџџџџџ
к
Btraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Sum_1SumEtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Select_1Ttraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
_output_shapes
:
О
Ftraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Reshape_1ReshapeBtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Sum_1Dtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape_1*
_output_shapes
: *
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
Tshape0
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
Ltraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape_2ShapeDtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Reshape*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
out_type0*
_output_shapes
:*
T0
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
Jtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/zerosFillLtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape_2Ptraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/zeros/Const*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*

index_type0
ћ
Ntraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/LessEqual	LessEqualloss/dense_3_loss/truedivloss/dense_3_loss/sub*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ

Ztraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/ShapeLtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum
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
Htraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/SumSumKtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/SelectZtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
щ
Ltraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/ReshapeReshapeHtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/SumJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
Tshape0
њ
Jtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Sum_1SumMtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Select_1\training/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum
о
Ntraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Reshape_1ReshapeJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Sum_1Ltraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape_1*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
Tshape0*
_output_shapes
: 
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
:training/Adam/gradients/loss/dense_3_loss/truediv_grad/SumSum>training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDivLtraining/Adam/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
:
Б
>training/Adam/gradients/loss/dense_3_loss/truediv_grad/ReshapeReshape:training/Adam/gradients/loss/dense_3_loss/truediv_grad/Sum<training/Adam/gradients/loss/dense_3_loss/truediv_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*
Tshape0
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
<training/Adam/gradients/loss/dense_3_loss/truediv_grad/Sum_1Sum:training/Adam/gradients/loss/dense_3_loss/truediv_grad/mulNtraining/Adam/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv
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
7training/Adam/gradients/loss/dense_3_loss/Sum_grad/SizeConst*
dtype0*
_output_shapes
: *(
_class
loc:@loss/dense_3_loss/Sum*
value	B :
ъ
6training/Adam/gradients/loss/dense_3_loss/Sum_grad/addAdd'loss/dense_3_loss/Sum/reduction_indices7training/Adam/gradients/loss/dense_3_loss/Sum_grad/Size*
T0*(
_class
loc:@loss/dense_3_loss/Sum*
_output_shapes
: 
ў
6training/Adam/gradients/loss/dense_3_loss/Sum_grad/modFloorMod6training/Adam/gradients/loss/dense_3_loss/Sum_grad/add7training/Adam/gradients/loss/dense_3_loss/Sum_grad/Size*
_output_shapes
: *
T0*(
_class
loc:@loss/dense_3_loss/Sum
Ї
:training/Adam/gradients/loss/dense_3_loss/Sum_grad/Shape_1Const*(
_class
loc:@loss/dense_3_loss/Sum*
valueB *
dtype0*
_output_shapes
: 
Њ
>training/Adam/gradients/loss/dense_3_loss/Sum_grad/range/startConst*(
_class
loc:@loss/dense_3_loss/Sum*
value	B : *
dtype0*
_output_shapes
: 
Њ
>training/Adam/gradients/loss/dense_3_loss/Sum_grad/range/deltaConst*(
_class
loc:@loss/dense_3_loss/Sum*
value	B :*
dtype0*
_output_shapes
: 
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
7training/Adam/gradients/loss/dense_3_loss/Sum_grad/FillFill:training/Adam/gradients/loss/dense_3_loss/Sum_grad/Shape_1=training/Adam/gradients/loss/dense_3_loss/Sum_grad/Fill/value*
T0*(
_class
loc:@loss/dense_3_loss/Sum*

index_type0*
_output_shapes
: 

@training/Adam/gradients/loss/dense_3_loss/Sum_grad/DynamicStitchDynamicStitch8training/Adam/gradients/loss/dense_3_loss/Sum_grad/range6training/Adam/gradients/loss/dense_3_loss/Sum_grad/mod8training/Adam/gradients/loss/dense_3_loss/Sum_grad/Shape7training/Adam/gradients/loss/dense_3_loss/Sum_grad/Fill*
T0*(
_class
loc:@loss/dense_3_loss/Sum*
N*
_output_shapes
:
Ј
<training/Adam/gradients/loss/dense_3_loss/Sum_grad/Maximum/yConst*
_output_shapes
: *(
_class
loc:@loss/dense_3_loss/Sum*
value	B :*
dtype0

:training/Adam/gradients/loss/dense_3_loss/Sum_grad/MaximumMaximum@training/Adam/gradients/loss/dense_3_loss/Sum_grad/DynamicStitch<training/Adam/gradients/loss/dense_3_loss/Sum_grad/Maximum/y*
_output_shapes
:*
T0*(
_class
loc:@loss/dense_3_loss/Sum

;training/Adam/gradients/loss/dense_3_loss/Sum_grad/floordivFloorDiv8training/Adam/gradients/loss/dense_3_loss/Sum_grad/Shape:training/Adam/gradients/loss/dense_3_loss/Sum_grad/Maximum*
T0*(
_class
loc:@loss/dense_3_loss/Sum*
_output_shapes
:
М
:training/Adam/gradients/loss/dense_3_loss/Sum_grad/ReshapeReshape@training/Adam/gradients/loss/dense_3_loss/truediv_grad/Reshape_1@training/Adam/gradients/loss/dense_3_loss/Sum_grad/DynamicStitch*
T0*(
_class
loc:@loss/dense_3_loss/Sum*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
І
7training/Adam/gradients/loss/dense_3_loss/Sum_grad/TileTile:training/Adam/gradients/loss/dense_3_loss/Sum_grad/Reshape;training/Adam/gradients/loss/dense_3_loss/Sum_grad/floordiv*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0*(
_class
loc:@loss/dense_3_loss/Sum

training/Adam/gradients/AddNAddN>training/Adam/gradients/loss/dense_3_loss/truediv_grad/Reshape7training/Adam/gradients/loss/dense_3_loss/Sum_grad/Tile*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*
N*'
_output_shapes
:џџџџџџџџџ
М
0training/Adam/gradients/dense_3/Softmax_grad/mulMultraining/Adam/gradients/AddNdense_3/Softmax*'
_output_shapes
:џџџџџџџџџ*
T0*"
_class
loc:@dense_3/Softmax
Б
Btraining/Adam/gradients/dense_3/Softmax_grad/Sum/reduction_indicesConst*
_output_shapes
: *"
_class
loc:@dense_3/Softmax*
valueB :
џџџџџџџџџ*
dtype0
 
0training/Adam/gradients/dense_3/Softmax_grad/SumSum0training/Adam/gradients/dense_3/Softmax_grad/mulBtraining/Adam/gradients/dense_3/Softmax_grad/Sum/reduction_indices*
	keep_dims(*

Tidx0*
T0*"
_class
loc:@dense_3/Softmax*'
_output_shapes
:џџџџџџџџџ
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
2training/Adam/gradients/dense_3/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_3/Softmax_grad/mul_1dense_3/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_3/MatMul
ћ
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldropout_2/cond/Merge2training/Adam/gradients/dense_3/Softmax_grad/mul_1*!
_class
loc:@dense_3/MatMul*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0
ћ
;training/Adam/gradients/dropout_2/cond/Merge_grad/cond_gradSwitch2training/Adam/gradients/dense_3/MatMul_grad/MatMuldropout_2/cond/pred_id*!
_class
loc:@dense_3/MatMul*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*
T0
И
 training/Adam/gradients/Switch_2Switchdense_2/Reludropout_2/cond/pred_id*
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
training/Adam/gradients/Shape_1Shape"training/Adam/gradients/Switch_2:1*
_output_shapes
:*
T0*
_class
loc:@dense_2/Relu*
out_type0
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
>training/Adam/gradients/dropout_2/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_2/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros*
T0*
_class
loc:@dense_2/Relu*
N**
_output_shapes
:џџџџџџџџџ: 
Ц
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ShapeShapedropout_2/cond/dropout/div*-
_class#
!loc:@dropout_2/cond/dropout/mul*
out_type0*
_output_shapes
:*
T0
Ъ
?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1Shapedropout_2/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
out_type0*
_output_shapes
:
в
Mtraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/MulMul=training/Adam/gradients/dropout_2/cond/Merge_grad/cond_grad:1dropout_2/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*(
_output_shapes
:џџџџџџџџџ
Н
;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
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
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
_output_shapes
:
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
?training/Adam/gradients/dropout_2/cond/dropout/div_grad/Shape_1Const*
_output_shapes
: *-
_class#
!loc:@dropout_2/cond/dropout/div*
valueB *
dtype0
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
;training/Adam/gradients/dropout_2/cond/dropout/div_grad/mulMul?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ReshapeAtraining/Adam/gradients/dropout_2/cond/dropout/div_grad/RealDiv_2*(
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_2/cond/dropout/div
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
Atraining/Adam/gradients/dropout_2/cond/dropout/div_grad/Reshape_1Reshape=training/Adam/gradients/dropout_2/cond/dropout/div_grad/Sum_1?training/Adam/gradients/dropout_2/cond/dropout/div_grad/Shape_1*
_output_shapes
: *
T0*-
_class#
!loc:@dropout_2/cond/dropout/div*
Tshape0
З
5training/Adam/gradients/dropout_2/cond/mul_grad/ShapeShapedropout_2/cond/mul/Switch:1*
T0*%
_class
loc:@dropout_2/cond/mul*
out_type0*
_output_shapes
:
Ё
7training/Adam/gradients/dropout_2/cond/mul_grad/Shape_1Const*%
_class
loc:@dropout_2/cond/mul*
valueB *
dtype0*
_output_shapes
: 
В
Etraining/Adam/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/dropout_2/cond/mul_grad/Shape7training/Adam/gradients/dropout_2/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_2/cond/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ы
3training/Adam/gradients/dropout_2/cond/mul_grad/MulMul?training/Adam/gradients/dropout_2/cond/dropout/div_grad/Reshapedropout_2/cond/mul/y*(
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@dropout_2/cond/mul

3training/Adam/gradients/dropout_2/cond/mul_grad/SumSum3training/Adam/gradients/dropout_2/cond/mul_grad/MulEtraining/Adam/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgs*%
_class
loc:@dropout_2/cond/mul*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

7training/Adam/gradients/dropout_2/cond/mul_grad/ReshapeReshape3training/Adam/gradients/dropout_2/cond/mul_grad/Sum5training/Adam/gradients/dropout_2/cond/mul_grad/Shape*
T0*%
_class
loc:@dropout_2/cond/mul*
Tshape0*(
_output_shapes
:џџџџџџџџџ
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
9training/Adam/gradients/dropout_2/cond/mul_grad/Reshape_1Reshape5training/Adam/gradients/dropout_2/cond/mul_grad/Sum_17training/Adam/gradients/dropout_2/cond/mul_grad/Shape_1*%
_class
loc:@dropout_2/cond/mul*
Tshape0*
_output_shapes
: *
T0
И
 training/Adam/gradients/Switch_3Switchdense_2/Reludropout_2/cond/pred_id*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*
T0*
_class
loc:@dense_2/Relu
Є
"training/Adam/gradients/Identity_1Identity training/Adam/gradients/Switch_3*
T0*
_class
loc:@dense_2/Relu*(
_output_shapes
:џџџџџџџџџ
 
training/Adam/gradients/Shape_2Shape training/Adam/gradients/Switch_3*
T0*
_class
loc:@dense_2/Relu*
out_type0*
_output_shapes
:
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
@training/Adam/gradients/dropout_2/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_17training/Adam/gradients/dropout_2/cond/mul_grad/Reshape*
T0*
_class
loc:@dense_2/Relu*
N**
_output_shapes
:џџџџџџџџџ: 

training/Adam/gradients/AddN_1AddN>training/Adam/gradients/dropout_2/cond/Switch_1_grad/cond_grad@training/Adam/gradients/dropout_2/cond/mul/Switch_grad/cond_grad*
_class
loc:@dense_2/Relu*
N*(
_output_shapes
:џџџџџџџџџ*
T0
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
2training/Adam/gradients/dense_2/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_2/Relu_grad/ReluGraddense_2/kernel/read*!
_class
loc:@dense_2/MatMul*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
ќ
4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1MatMuldropout_1/cond/Merge2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
T0*!
_class
loc:@dense_2/MatMul* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
ћ
;training/Adam/gradients/dropout_1/cond/Merge_grad/cond_gradSwitch2training/Adam/gradients/dense_2/MatMul_grad/MatMuldropout_1/cond/pred_id*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*
T0*!
_class
loc:@dense_2/MatMul
И
 training/Adam/gradients/Switch_4Switchdense_1/Reludropout_1/cond/pred_id*
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
%training/Adam/gradients/zeros_2/ConstConst#^training/Adam/gradients/Identity_2*
_output_shapes
: *
_class
loc:@dense_1/Relu*
valueB
 *    *
dtype0
е
training/Adam/gradients/zeros_2Filltraining/Adam/gradients/Shape_3%training/Adam/gradients/zeros_2/Const*
_class
loc:@dense_1/Relu*

index_type0*(
_output_shapes
:џџџџџџџџџ*
T0

>training/Adam/gradients/dropout_1/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_1/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros_2**
_output_shapes
:џџџџџџџџџ: *
T0*
_class
loc:@dense_1/Relu*
N
Ц
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ShapeShapedropout_1/cond/dropout/div*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
out_type0*
_output_shapes
:
Ъ
?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1Shapedropout_1/cond/dropout/Floor*
_output_shapes
:*
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

;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/MulMul=training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad:1dropout_1/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*(
_output_shapes
:џџџџџџџџџ
Н
;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
Tshape0

=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Muldropout_1/cond/dropout/div=training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad:1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*(
_output_shapes
:џџџџџџџџџ
У
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul
М
Atraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
Tshape0*(
_output_shapes
:џџџџџџџџџ
О
=training/Adam/gradients/dropout_1/cond/dropout/div_grad/ShapeShapedropout_1/cond/mul*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*
out_type0*
_output_shapes
:
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

?training/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDivRealDiv?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Reshape dropout_1/cond/dropout/keep_prob*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ
С
;training/Adam/gradients/dropout_1/cond/dropout/div_grad/SumSum?training/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDivMtraining/Adam/gradients/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div
Ж
?training/Adam/gradients/dropout_1/cond/dropout/div_grad/ReshapeReshape;training/Adam/gradients/dropout_1/cond/dropout/div_grad/Sum=training/Adam/gradients/dropout_1/cond/dropout/div_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*
Tshape0
И
;training/Adam/gradients/dropout_1/cond/dropout/div_grad/NegNegdropout_1/cond/mul*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ

Atraining/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDiv_1RealDiv;training/Adam/gradients/dropout_1/cond/dropout/div_grad/Neg dropout_1/cond/dropout/keep_prob*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ

Atraining/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDiv_2RealDivAtraining/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDiv_1 dropout_1/cond/dropout/keep_prob*(
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div
Ј
;training/Adam/gradients/dropout_1/cond/dropout/div_grad/mulMul?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ReshapeAtraining/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDiv_2*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ
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
Atraining/Adam/gradients/dropout_1/cond/dropout/div_grad/Reshape_1Reshape=training/Adam/gradients/dropout_1/cond/dropout/div_grad/Sum_1?training/Adam/gradients/dropout_1/cond/dropout/div_grad/Shape_1*-
_class#
!loc:@dropout_1/cond/dropout/div*
Tshape0*
_output_shapes
: *
T0
З
5training/Adam/gradients/dropout_1/cond/mul_grad/ShapeShapedropout_1/cond/mul/Switch:1*%
_class
loc:@dropout_1/cond/mul*
out_type0*
_output_shapes
:*
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
ы
3training/Adam/gradients/dropout_1/cond/mul_grad/MulMul?training/Adam/gradients/dropout_1/cond/dropout/div_grad/Reshapedropout_1/cond/mul/y*
T0*%
_class
loc:@dropout_1/cond/mul*(
_output_shapes
:џџџџџџџџџ

3training/Adam/gradients/dropout_1/cond/mul_grad/SumSum3training/Adam/gradients/dropout_1/cond/mul_grad/MulEtraining/Adam/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgs*
T0*%
_class
loc:@dropout_1/cond/mul*
_output_shapes
:*
	keep_dims( *

Tidx0

7training/Adam/gradients/dropout_1/cond/mul_grad/ReshapeReshape3training/Adam/gradients/dropout_1/cond/mul_grad/Sum5training/Adam/gradients/dropout_1/cond/mul_grad/Shape*
T0*%
_class
loc:@dropout_1/cond/mul*
Tshape0*(
_output_shapes
:џџџџџџџџџ
є
5training/Adam/gradients/dropout_1/cond/mul_grad/Mul_1Muldropout_1/cond/mul/Switch:1?training/Adam/gradients/dropout_1/cond/dropout/div_grad/Reshape*
T0*%
_class
loc:@dropout_1/cond/mul*(
_output_shapes
:џџџџџџџџџ
Ѓ
5training/Adam/gradients/dropout_1/cond/mul_grad/Sum_1Sum5training/Adam/gradients/dropout_1/cond/mul_grad/Mul_1Gtraining/Adam/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@dropout_1/cond/mul*
_output_shapes
:

9training/Adam/gradients/dropout_1/cond/mul_grad/Reshape_1Reshape5training/Adam/gradients/dropout_1/cond/mul_grad/Sum_17training/Adam/gradients/dropout_1/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_1/cond/mul*
Tshape0*
_output_shapes
: 
И
 training/Adam/gradients/Switch_5Switchdense_1/Reludropout_1/cond/pred_id*
_class
loc:@dense_1/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*
T0
Є
"training/Adam/gradients/Identity_3Identity training/Adam/gradients/Switch_5*
_class
loc:@dense_1/Relu*(
_output_shapes
:џџџџџџџџџ*
T0
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
@training/Adam/gradients/dropout_1/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_37training/Adam/gradients/dropout_1/cond/mul_grad/Reshape*
T0*
_class
loc:@dense_1/Relu*
N**
_output_shapes
:џџџџџџџџџ: 

training/Adam/gradients/AddN_2AddN>training/Adam/gradients/dropout_1/cond/Switch_1_grad/cond_grad@training/Adam/gradients/dropout_1/cond/mul/Switch_grad/cond_grad*
T0*
_class
loc:@dense_1/Relu*
N*(
_output_shapes
:џџџџџџџџџ
Р
2training/Adam/gradients/dense_1/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_2dense_1/Relu*
T0*
_class
loc:@dense_1/Relu*(
_output_shapes
:џџџџџџџџџ
м
8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_1/Relu_grad/ReluGrad*"
_class
loc:@dense_1/BiasAdd*
data_formatNHWC*
_output_shapes	
:*
T0

2training/Adam/gradients/dense_1/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_1/Relu_grad/ReluGraddense_1/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_1/MatMul
љ
4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1MatMulflatten_1/Reshape2training/Adam/gradients/dense_1/Relu_grad/ReluGrad*
T0*!
_class
loc:@dense_1/MatMul* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
Ќ
4training/Adam/gradients/flatten_1/Reshape_grad/ShapeShapelstm_1/transpose_1*
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
 :џџџџџџџџџџџџџџџџџџ
Л
Atraining/Adam/gradients/lstm_1/transpose_1_grad/InvertPermutationInvertPermutationlstm_1/transpose_1/perm*
_output_shapes
:*
T0*%
_class
loc:@lstm_1/transpose_1
Д
9training/Adam/gradients/lstm_1/transpose_1_grad/transpose	Transpose6training/Adam/gradients/flatten_1/Reshape_grad/ReshapeAtraining/Adam/gradients/lstm_1/transpose_1_grad/InvertPermutation*
T0*%
_class
loc:@lstm_1/transpose_1*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
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
:џџџџџџџџџ
x
$training/Adam/gradients/zeros_like_1	ZerosLikelstm_1/while/Exit_4*
T0*'
_output_shapes
:џџџџџџџџџ
э
7training/Adam/gradients/lstm_1/while/Exit_2_grad/b_exitEnterptraining/Adam/gradients/lstm_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*&
_class
loc:@lstm_1/while/Exit_2*
parallel_iterations *
is_constant( *
_output_shapes
: *B

frame_name42training/Adam/gradients/lstm_1/while/while_context
А
7training/Adam/gradients/lstm_1/while/Exit_3_grad/b_exitEnter"training/Adam/gradients/zeros_like*
is_constant( *'
_output_shapes
:џџџџџџџџџ*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*&
_class
loc:@lstm_1/while/Exit_3*
parallel_iterations 
В
7training/Adam/gradients/lstm_1/while/Exit_4_grad/b_exitEnter$training/Adam/gradients/zeros_like_1*
is_constant( *'
_output_shapes
:џџџџџџџџџ*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*&
_class
loc:@lstm_1/while/Exit_4*
parallel_iterations 
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
:џџџџџџџџџ: 
Ї
;training/Adam/gradients/lstm_1/while/Switch_4_grad/b_switchMerge7training/Adam/gradients/lstm_1/while/Exit_4_grad/b_exitBtraining/Adam/gradients/lstm_1/while/Switch_4_grad_1/NextIteration*)
_output_shapes
:џџџџџџџџџ: *
T0*'
_class
loc:@lstm_1/while/Merge_4*
N
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
&:џџџџџџџџџ:џџџџџџџџџ

8training/Adam/gradients/lstm_1/while/Merge_4_grad/SwitchSwitch;training/Adam/gradients/lstm_1/while/Switch_4_grad/b_switch!training/Adam/gradients/b_count_2*
T0*'
_class
loc:@lstm_1/while/Merge_4*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
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
:џџџџџџџџџ
г
6training/Adam/gradients/lstm_1/while/Enter_4_grad/ExitExit8training/Adam/gradients/lstm_1/while/Merge_4_grad/Switch*
T0*'
_class
loc:@lstm_1/while/Enter_4*'
_output_shapes
:џџџџџџџџџ
Ё
otraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3utraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter:training/Adam/gradients/lstm_1/while/Merge_2_grad/Switch:1*
_output_shapes

:: *%
_class
loc:@lstm_1/while/mul_5*#
sourcetraining/Adam/gradients
а
utraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlstm_1/TensorArray*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations 
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
:џџџџџџџџџ
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
ktraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2etraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterlstm_1/while/Identity_1^training/Adam/gradients/Add*
_output_shapes
: *
swap_memory(*
T0*%
_class
loc:@lstm_1/while/mul_5
и
jtraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2ptraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^training/Adam/gradients/Sub*
_output_shapes
: *
	elem_type0*%
_class
loc:@lstm_1/while/mul_5

ptraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnteretraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5
р
ftraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerG^training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV2G^training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV2k^training/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Q^training/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2_1Q^training/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopV2Q^training/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2_1Q^training/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopV2Q^training/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2_1Q^training/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2_1Q^training/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2_1Q^training/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPopV2a^training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopV2U^training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopV2c^training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopV2W^training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopV2[^training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopV2R^training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopV2c^training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopV2W^training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopV2[^training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopV2R^training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopV2Y^training/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopV2P^training/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopV2Q^training/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopV2?^training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPopV2Q^training/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2_1?^training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPopV2A^training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPopV2Q^training/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2_1?^training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPopV2A^training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPopV2Q^training/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopV2?^training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPopV2Q^training/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2_1?^training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV2A^training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPopV2O^training/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopV2=^training/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPopV2*%
_class
loc:@lstm_1/while/mul_5
Ї
training/Adam/gradients/AddN_3AddN:training/Adam/gradients/lstm_1/while/Merge_3_grad/Switch:1_training/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
T0*'
_class
loc:@lstm_1/while/Merge_3*
N*'
_output_shapes
:џџџџџџџџџ
И
5training/Adam/gradients/lstm_1/while/mul_5_grad/ShapeShapelstm_1/while/clip_by_value_2*
_output_shapes
:*
T0*%
_class
loc:@lstm_1/while/mul_5*
out_type0
Б
7training/Adam/gradients/lstm_1/while/mul_5_grad/Shape_1Shapelstm_1/while/Tanh_1*
T0*%
_class
loc:@lstm_1/while/mul_5*
out_type0*
_output_shapes
:
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
Ktraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Const*
	elem_type0*%
_class
loc:@lstm_1/while/mul_5*

stack_name *
_output_shapes
:
Ч
Ktraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations 
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
Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Const_1*

stack_name *
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/mul_5
Ы
Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc_1*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
ч
Straining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/mul_5_grad/Shape_1^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
:*
swap_memory(
Ќ
Rtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
:*
	elem_type0
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
3training/Adam/gradients/lstm_1/while/mul_5_grad/MulMultraining/Adam/gradients/AddN_3>training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/mul_5
Х
9training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/ConstConst*
dtype0*
_output_shapes
: *?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_5*
valueB :
џџџџџџџџџ
џ
9training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/f_accStackV29training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/Const*?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_5*

stack_name *
_output_shapes
:*
	elem_type0
Ѓ
9training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/EnterEnter9training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations 
Ј
?training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPushV2StackPushV29training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/Enterlstm_1/while/Tanh_1^training/Adam/gradients/Add*%
_class
loc:@lstm_1/while/mul_5*'
_output_shapes
:џџџџџџџџџ*
swap_memory(*
T0

>training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV2
StackPopV2Dtraining/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_5*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
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
3training/Adam/gradients/lstm_1/while/mul_5_grad/SumSum3training/Adam/gradients/lstm_1/while/mul_5_grad/MulEtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
:
А
7training/Adam/gradients/lstm_1/while/mul_5_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/mul_5_grad/SumPtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/mul_5*
Tshape0
ї
5training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1Mul@training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPopV2training/Adam/gradients/AddN_3*
T0*%
_class
loc:@lstm_1/while/mul_5*'
_output_shapes
:џџџџџџџџџ
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
;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/f_accStackV2;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/Const*H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_5*

stack_name *
_output_shapes
:*
	elem_type0
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
Atraining/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPushV2StackPushV2;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/Enterlstm_1/while/clip_by_value_2^training/Adam/gradients/Add*%
_class
loc:@lstm_1/while/mul_5*'
_output_shapes
:џџџџџџџџџ*
swap_memory(*
T0

@training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPopV2
StackPopV2Ftraining/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_5*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
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
5training/Adam/gradients/lstm_1/while/mul_5_grad/Sum_1Sum5training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1Gtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ж
9training/Adam/gradients/lstm_1/while/mul_5_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/mul_5_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/mul_5*
Tshape0*'
_output_shapes
:џџџџџџџџџ
д
?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/ShapeShape$lstm_1/while/clip_by_value_2/Minimum*
_output_shapes
:*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
out_type0
г
Atraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Shape_1Const^training/Adam/gradients/Sub*
dtype0*
_output_shapes
: */
_class%
#!loc:@lstm_1/while/clip_by_value_2*
valueB 
щ
Atraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Shape_2Shape7training/Adam/gradients/lstm_1/while/mul_5_grad/Reshape*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
out_type0*
_output_shapes
:*
T0
й
Etraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/zeros/ConstConst^training/Adam/gradients/Sub*
dtype0*
_output_shapes
: */
_class%
#!loc:@lstm_1/while/clip_by_value_2*
valueB
 *    
Ц
?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/zerosFillAtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Shape_2Etraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/zeros/Const*'
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*

index_type0
м
Ftraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqualGreaterEqualQtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopV2Ntraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/Const_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*'
_output_shapes
:џџџџџџџџџ*
T0
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
Rtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPushV2StackPushV2Ltraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/Enter$lstm_1/while/clip_by_value_2/Minimum^training/Adam/gradients/Add*'
_output_shapes
:џџџџџџџџџ*
swap_memory(*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2
С
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopV2
StackPopV2Wtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopV2/Enter^training/Adam/gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
і
Wtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopV2/EnterEnterLtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2
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
Otraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgsBroadcastGradientArgsZtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopV2Atraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Shape_1*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
б
Utraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: */
_class%
#!loc:@lstm_1/while/clip_by_value_2*
valueB :
џџџџџџџџџ
Ї
Utraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_accStackV2Utraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/Const*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*

stack_name *
_output_shapes
:*
	elem_type0
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
@training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/SelectSelectFtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual7training/Adam/gradients/lstm_1/while/mul_5_grad/Reshape?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/zeros*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*'
_output_shapes
:џџџџџџџџџ
ё
Btraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Select_1SelectFtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/zeros7training/Adam/gradients/lstm_1/while/mul_5_grad/Reshape*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*'
_output_shapes
:џџџџџџџџџ
Ш
=training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/SumSum@training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/SelectOtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
_output_shapes
:
и
Atraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/ReshapeReshape=training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/SumZtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopV2*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Sum_1SumBtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Select_1Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2
В
Ctraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Reshape_1Reshape?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Sum_1Atraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Shape_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
Tshape0*
_output_shapes
: *
T0

9training/Adam/gradients/lstm_1/while/Tanh_1_grad/TanhGradTanhGrad>training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV29training/Adam/gradients/lstm_1/while/mul_5_grad/Reshape_1*
T0*&
_class
loc:@lstm_1/while/Tanh_1*'
_output_shapes
:џџџџџџџџџ
й
Btraining/Adam/gradients/lstm_1/while/Switch_2_grad_1/NextIterationNextIteration:training/Adam/gradients/lstm_1/while/Merge_2_grad/Switch:1*
T0*'
_class
loc:@lstm_1/while/Merge_2*
_output_shapes
: 
в
Gtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ShapeShapelstm_1/while/add_8*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
out_type0*
_output_shapes
:*
T0
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
:џџџџџџџџџ
№
Ktraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual	LessEqualVtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopV2Straining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/Const_1*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*'
_output_shapes
:џџџџџџџџџ
ю
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/ConstConst*
dtype0*
_output_shapes
: *P
_classF
Dloc:@lstm_1/while/add_8)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB :
џџџџџџџџџ
Р
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_accStackV2Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/Const*
	elem_type0*P
_classF
Dloc:@lstm_1/while/add_8)loc:@lstm_1/while/clip_by_value_2/Minimum*

stack_name *
_output_shapes
:
х
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/EnterEnterQtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
щ
Wtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPushV2StackPushV2Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/Enterlstm_1/while/add_8^training/Adam/gradients/Add*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*'
_output_shapes
:џџџџџџџџџ*
swap_memory(
г
Vtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopV2
StackPopV2\training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopV2/Enter^training/Adam/gradients/Sub*'
_output_shapes
:џџџџџџџџџ*
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
Wtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsbtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopV2Itraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum
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
]training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/EnterEnter]training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_acc*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
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
Htraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SelectSelectKtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqualAtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/ReshapeGtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros*'
_output_shapes
:џџџџџџџџџ*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum

Jtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Select_1SelectKtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqualGtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zerosAtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Reshape*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*'
_output_shapes
:џџџџџџџџџ
ш
Etraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SumSumHtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SelectWtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ј
Itraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ReshapeReshapeEtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Sumbtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopV2*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
Tshape0*'
_output_shapes
:џџџџџџџџџ
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
Ktraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Reshape_1ReshapeGtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Sum_1Itraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_1*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
Tshape0*
_output_shapes
: 

training/Adam/gradients/AddN_4AddN:training/Adam/gradients/lstm_1/while/Merge_4_grad/Switch:19training/Adam/gradients/lstm_1/while/Tanh_1_grad/TanhGrad*'
_output_shapes
:џџџџџџџџџ*
T0*'
_class
loc:@lstm_1/while/Merge_4*
N
Ў
5training/Adam/gradients/lstm_1/while/add_6_grad/ShapeShapelstm_1/while/mul_2*
_output_shapes
:*
T0*%
_class
loc:@lstm_1/while/add_6*
out_type0
А
7training/Adam/gradients/lstm_1/while/add_6_grad/Shape_1Shapelstm_1/while/mul_3*
T0*%
_class
loc:@lstm_1/while/add_6*
out_type0*
_output_shapes
:
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
Ktraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc*%
_class
loc:@lstm_1/while/add_6*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0
с
Qtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/add_6_grad/Shape^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/add_6*
_output_shapes
:*
swap_memory(
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
Straining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/add_6_grad/Shape_1^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/add_6*
_output_shapes
:*
swap_memory(
Ќ
Rtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_6*
_output_shapes
:*
	elem_type0
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
3training/Adam/gradients/lstm_1/while/add_6_grad/SumSumtraining/Adam/gradients/AddN_4Etraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_6
А
7training/Adam/gradients/lstm_1/while/add_6_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_6_grad/SumPtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/add_6*
Tshape0*'
_output_shapes
:џџџџџџџџџ

5training/Adam/gradients/lstm_1/while/add_6_grad/Sum_1Sumtraining/Adam/gradients/AddN_4Gtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs:1*
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
:џџџџџџџџџ
Ў
5training/Adam/gradients/lstm_1/while/add_8_grad/ShapeShapelstm_1/while/mul_4*
T0*%
_class
loc:@lstm_1/while/add_8*
out_type0*
_output_shapes
:
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
Ktraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/add_8*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/f_acc*
T0*%
_class
loc:@lstm_1/while/add_8*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
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
3training/Adam/gradients/lstm_1/while/add_8_grad/SumSumItraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ReshapeEtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_8*
_output_shapes
:
А
7training/Adam/gradients/lstm_1/while/add_8_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_8_grad/SumPtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/add_8*
Tshape0*'
_output_shapes
:џџџџџџџџџ
З
5training/Adam/gradients/lstm_1/while/add_8_grad/Sum_1SumItraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ReshapeGtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_8

9training/Adam/gradients/lstm_1/while/add_8_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_8_grad/Sum_17training/Adam/gradients/lstm_1/while/add_8_grad/Shape_1*
_output_shapes
: *
T0*%
_class
loc:@lstm_1/while/add_8*
Tshape0
И
5training/Adam/gradients/lstm_1/while/mul_2_grad/ShapeShapelstm_1/while/clip_by_value_1*
_output_shapes
:*
T0*%
_class
loc:@lstm_1/while/mul_2*
out_type0
Е
7training/Adam/gradients/lstm_1/while/mul_2_grad/Shape_1Shapelstm_1/while/Identity_4*
_output_shapes
:*
T0*%
_class
loc:@lstm_1/while/mul_2*
out_type0
ш
Etraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2Rtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/mul_2*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *%
_class
loc:@lstm_1/while/mul_2*
valueB :
џџџџџџџџџ
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
Vtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_2
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
Mtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/Const_1*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/mul_2*

stack_name 
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
Straining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/mul_2_grad/Shape_1^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_2*
_output_shapes
:*
swap_memory(
Ќ
Rtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*
	elem_type0*%
_class
loc:@lstm_1/while/mul_2*
_output_shapes
:
ю
Xtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterMtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc_1*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations 

3training/Adam/gradients/lstm_1/while/mul_2_grad/MulMul7training/Adam/gradients/lstm_1/while/add_6_grad/Reshape>training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_2*'
_output_shapes
:џџџџџџџџџ
Щ
9training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/ConstConst*
dtype0*
_output_shapes
: *C
_class9
7loc:@lstm_1/while/Identity_4loc:@lstm_1/while/mul_2*
valueB :
џџџџџџџџџ

9training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/f_accStackV29training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/Const*C
_class9
7loc:@lstm_1/while/Identity_4loc:@lstm_1/while/mul_2*

stack_name *
_output_shapes
:*
	elem_type0
Ѓ
9training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/EnterEnter9training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/f_acc*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0
Ќ
?training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPushV2StackPushV29training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/Enterlstm_1/while/Identity_4^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_2*'
_output_shapes
:џџџџџџџџџ*
swap_memory(

>training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPopV2
StackPopV2Dtraining/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_2*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
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
3training/Adam/gradients/lstm_1/while/mul_2_grad/SumSum3training/Adam/gradients/lstm_1/while/mul_2_grad/MulEtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs*
T0*%
_class
loc:@lstm_1/while/mul_2*
_output_shapes
:*
	keep_dims( *

Tidx0
А
7training/Adam/gradients/lstm_1/while/mul_2_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/mul_2_grad/SumPtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_2*
Tshape0*'
_output_shapes
:џџџџџџџџџ

5training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1Mul@training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPopV27training/Adam/gradients/lstm_1/while/add_6_grad/Reshape*
T0*%
_class
loc:@lstm_1/while/mul_2*'
_output_shapes
:џџџџџџџџџ
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
;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/f_accStackV2;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/Const*
_output_shapes
:*
	elem_type0*H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_2*

stack_name 
Ї
;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/EnterEnter;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations 
Е
Atraining/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPushV2StackPushV2;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/Enterlstm_1/while/clip_by_value_1^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_2*'
_output_shapes
:џџџџџџџџџ*
swap_memory(

@training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPopV2
StackPopV2Ftraining/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPopV2/Enter^training/Adam/gradients/Sub*'
_output_shapes
:џџџџџџџџџ*
	elem_type0*%
_class
loc:@lstm_1/while/mul_2
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
9training/Adam/gradients/lstm_1/while/mul_2_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/mul_2_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*%
_class
loc:@lstm_1/while/mul_2*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
Ж
5training/Adam/gradients/lstm_1/while/mul_3_grad/ShapeShapelstm_1/while/clip_by_value*
T0*%
_class
loc:@lstm_1/while/mul_3*
out_type0*
_output_shapes
:
Џ
7training/Adam/gradients/lstm_1/while/mul_3_grad/Shape_1Shapelstm_1/while/Tanh*
T0*%
_class
loc:@lstm_1/while/mul_3*
out_type0*
_output_shapes
:
ш
Etraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2Rtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/mul_3*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *%
_class
loc:@lstm_1/while/mul_3*
valueB :
џџџџџџџџџ

Ktraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/mul_3*

stack_name *
_output_shapes
:*
	elem_type0
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
Qtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/mul_3_grad/Shape^training/Adam/gradients/Add*%
_class
loc:@lstm_1/while/mul_3*
_output_shapes
:*
swap_memory(*
T0
Ј
Ptraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/mul_3
ъ
Vtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
П
Mtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *%
_class
loc:@lstm_1/while/mul_3*
valueB :
џџџџџџџџџ
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
:џџџџџџџџџ
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
9training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/EnterEnter9training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations 
І
?training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPushV2StackPushV29training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/Enterlstm_1/while/Tanh^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_3*'
_output_shapes
:џџџџџџџџџ*
swap_memory(

>training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPopV2
StackPopV2Dtraining/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_3*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
Ц
Dtraining/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPopV2/EnterEnter9training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/f_acc*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations *
is_constant(
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
:џџџџџџџџџ

5training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1Mul@training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPopV29training/Adam/gradients/lstm_1/while/add_6_grad/Reshape_1*
T0*%
_class
loc:@lstm_1/while/mul_3*'
_output_shapes
:џџџџџџџџџ
Ю
;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/ConstConst*F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_3*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/f_accStackV2;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/Const*F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_3*

stack_name *
_output_shapes
:*
	elem_type0
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
:џџџџџџџџџ*
swap_memory(

@training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPopV2
StackPopV2Ftraining/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_3*'
_output_shapes
:џџџџџџџџџ*
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
5training/Adam/gradients/lstm_1/while/mul_3_grad/Sum_1Sum5training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1Gtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/mul_3
Ж
9training/Adam/gradients/lstm_1/while/mul_3_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/mul_3_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2_1*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/mul_3*
Tshape0
Н
5training/Adam/gradients/lstm_1/while/mul_4_grad/ShapeConst^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_4*
valueB *
dtype0*
_output_shapes
: 
А
7training/Adam/gradients/lstm_1/while/mul_4_grad/Shape_1Shapelstm_1/while/add_7*%
_class
loc:@lstm_1/while/mul_4*
out_type0*
_output_shapes
:*
T0
Ы
Etraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/lstm_1/while/mul_4_grad/ShapePtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopV2*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/mul_4
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
Qtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/Enter7training/Adam/gradients/lstm_1/while/mul_4_grad/Shape_1^training/Adam/gradients/Add*%
_class
loc:@lstm_1/while/mul_4*
_output_shapes
:*
swap_memory(*
T0
Ј
Ptraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_4*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_4*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context

3training/Adam/gradients/lstm_1/while/mul_4_grad/MulMul7training/Adam/gradients/lstm_1/while/add_8_grad/Reshape>training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_4*'
_output_shapes
:џџџџџџџџџ
Ф
9training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/ConstConst*
_output_shapes
: *>
_class4
2loc:@lstm_1/while/add_7loc:@lstm_1/while/mul_4*
valueB :
џџџџџџџџџ*
dtype0
ў
9training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/f_accStackV29training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/Const*>
_class4
2loc:@lstm_1/while/add_7loc:@lstm_1/while/mul_4*

stack_name *
_output_shapes
:*
	elem_type0
Ѓ
9training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/EnterEnter9training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_4*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
Ї
?training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPushV2StackPushV29training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/Enterlstm_1/while/add_7^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_4*'
_output_shapes
:џџџџџџџџџ*
swap_memory(

>training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPopV2
StackPopV2Dtraining/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*
	elem_type0*%
_class
loc:@lstm_1/while/mul_4*'
_output_shapes
:џџџџџџџџџ
Ц
Dtraining/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPopV2/EnterEnter9training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_4
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
7training/Adam/gradients/lstm_1/while/mul_4_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/mul_4_grad/Sum5training/Adam/gradients/lstm_1/while/mul_4_grad/Shape*
_output_shapes
: *
T0*%
_class
loc:@lstm_1/while/mul_4*
Tshape0

5training/Adam/gradients/lstm_1/while/mul_4_grad/Mul_1Mul;training/Adam/gradients/lstm_1/while/mul_4_grad/Mul_1/Const7training/Adam/gradients/lstm_1/while/add_8_grad/Reshape*
T0*%
_class
loc:@lstm_1/while/mul_4*'
_output_shapes
:џџџџџџџџџ
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
:џџџџџџџџџ
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
?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/zerosFillAtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Shape_2Etraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/zeros/Const*'
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*

index_type0
м
Ftraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqualGreaterEqualQtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopV2Ntraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/Const_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*'
_output_shapes
:џџџџџџџџџ*
T0
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
Ltraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_accStackV2Ltraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/Const*
	elem_type0*Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*

stack_name *
_output_shapes
:
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
:џџџџџџџџџ*
swap_memory(*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1
С
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopV2
StackPopV2Wtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopV2/Enter^training/Adam/gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
і
Wtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopV2/EnterEnterLtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
parallel_iterations *
is_constant(
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
Otraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgsBroadcastGradientArgsZtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopV2Atraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Shape_1*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
б
Utraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/ConstConst*
_output_shapes
: */
_class%
#!loc:@lstm_1/while/clip_by_value_1*
valueB :
џџџџџџџџџ*
dtype0
Ї
Utraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_accStackV2Utraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/Const*
_output_shapes
:*
	elem_type0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*

stack_name 
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
StackPopV2`training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
_output_shapes
:*
	elem_type0
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
:џџџџџџџџџ*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1
ё
Btraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Select_1SelectFtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/zeros7training/Adam/gradients/lstm_1/while/mul_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1
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
Atraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/ReshapeReshape=training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/SumZtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopV2*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Sum_1SumBtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Select_1Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs:1*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
В
Ctraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Reshape_1Reshape?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Sum_1Atraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Shape_1*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
Tshape0*
_output_shapes
: 
Ю
=training/Adam/gradients/lstm_1/while/clip_by_value_grad/ShapeShape"lstm_1/while/clip_by_value/Minimum*-
_class#
!loc:@lstm_1/while/clip_by_value*
out_type0*
_output_shapes
:*
T0
Я
?training/Adam/gradients/lstm_1/while/clip_by_value_grad/Shape_1Const^training/Adam/gradients/Sub*
_output_shapes
: *-
_class#
!loc:@lstm_1/while/clip_by_value*
valueB *
dtype0
х
?training/Adam/gradients/lstm_1/while/clip_by_value_grad/Shape_2Shape7training/Adam/gradients/lstm_1/while/mul_3_grad/Reshape*
_output_shapes
:*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
out_type0
е
Ctraining/Adam/gradients/lstm_1/while/clip_by_value_grad/zeros/ConstConst^training/Adam/gradients/Sub*
dtype0*
_output_shapes
: *-
_class#
!loc:@lstm_1/while/clip_by_value*
valueB
 *    
О
=training/Adam/gradients/lstm_1/while/clip_by_value_grad/zerosFill?training/Adam/gradients/lstm_1/while/clip_by_value_grad/Shape_2Ctraining/Adam/gradients/lstm_1/while/clip_by_value_grad/zeros/Const*-
_class#
!loc:@lstm_1/while/clip_by_value*

index_type0*'
_output_shapes
:џџџџџџџџџ*
T0
д
Dtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqualGreaterEqualOtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopV2Ltraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/Const_1*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*'
_output_shapes
:џџџџџџџџџ
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
Jtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_accStackV2Jtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/Const*
	elem_type0*V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*

stack_name *
_output_shapes
:
Э
Jtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/EnterEnterJtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
parallel_iterations 
с
Ptraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPushV2StackPushV2Jtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/Enter"lstm_1/while/clip_by_value/Minimum^training/Adam/gradients/Add*'
_output_shapes
:џџџџџџџџџ*
swap_memory(*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value
Л
Otraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopV2
StackPopV2Utraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopV2/Enter^training/Adam/gradients/Sub*-
_class#
!loc:@lstm_1/while/clip_by_value*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
№
Utraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopV2/EnterEnterJtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
parallel_iterations 
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
Straining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_accStackV2Straining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/Const*
	elem_type0*-
_class#
!loc:@lstm_1/while/clip_by_value*

stack_name *
_output_shapes
:
п
Straining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/EnterEnterStraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_acc*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
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
^training/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopV2/EnterEnterStraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_acc*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
ч
>training/Adam/gradients/lstm_1/while/clip_by_value_grad/SelectSelectDtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual7training/Adam/gradients/lstm_1/while/mul_3_grad/Reshape=training/Adam/gradients/lstm_1/while/clip_by_value_grad/zeros*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*'
_output_shapes
:џџџџџџџџџ
щ
@training/Adam/gradients/lstm_1/while/clip_by_value_grad/Select_1SelectDtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual=training/Adam/gradients/lstm_1/while/clip_by_value_grad/zeros7training/Adam/gradients/lstm_1/while/mul_3_grad/Reshape*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*'
_output_shapes
:џџџџџџџџџ
Р
;training/Adam/gradients/lstm_1/while/clip_by_value_grad/SumSum>training/Adam/gradients/lstm_1/while/clip_by_value_grad/SelectMtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
_output_shapes
:*
	keep_dims( *

Tidx0
а
?training/Adam/gradients/lstm_1/while/clip_by_value_grad/ReshapeReshape;training/Adam/gradients/lstm_1/while/clip_by_value_grad/SumXtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopV2*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
Tshape0*'
_output_shapes
:џџџџџџџџџ
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
Atraining/Adam/gradients/lstm_1/while/clip_by_value_grad/Reshape_1Reshape=training/Adam/gradients/lstm_1/while/clip_by_value_grad/Sum_1?training/Adam/gradients/lstm_1/while/clip_by_value_grad/Shape_1*
_output_shapes
: *
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
Tshape0

7training/Adam/gradients/lstm_1/while/Tanh_grad/TanhGradTanhGrad>training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPopV29training/Adam/gradients/lstm_1/while/mul_3_grad/Reshape_1*
T0*$
_class
loc:@lstm_1/while/Tanh*'
_output_shapes
:џџџџџџџџџ
В
5training/Adam/gradients/lstm_1/while/add_7_grad/ShapeShapelstm_1/while/BiasAdd_3*
_output_shapes
:*
T0*%
_class
loc:@lstm_1/while/add_7*
out_type0
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
Ktraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc*
T0*%
_class
loc:@lstm_1/while/add_7*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
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
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
	elem_type0*%
_class
loc:@lstm_1/while/add_7*
_output_shapes
:
ъ
Vtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_7*
parallel_iterations *
is_constant(
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
Mtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/Const_1*%
_class
loc:@lstm_1/while/add_7*

stack_name *
_output_shapes
:*
	elem_type0
Ы
Mtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_7*
parallel_iterations *
is_constant(
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
7training/Adam/gradients/lstm_1/while/add_7_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_7_grad/SumPtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/add_7*
Tshape0
Ї
5training/Adam/gradients/lstm_1/while/add_7_grad/Sum_1Sum9training/Adam/gradients/lstm_1/while/mul_4_grad/Reshape_1Gtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_7
Ж
9training/Adam/gradients/lstm_1/while/add_7_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_7_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/add_7*
Tshape0*'
_output_shapes
:џџџџџџџџџ
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
Gtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zerosFillItraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_2Mtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros/Const*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*

index_type0*'
_output_shapes
:џџџџџџџџџ
№
Ktraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual	LessEqualVtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopV2Straining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/Const_1*'
_output_shapes
:џџџџџџџџџ*
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
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_accStackV2Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/Const*P
_classF
Dloc:@lstm_1/while/add_4)loc:@lstm_1/while/clip_by_value_1/Minimum*

stack_name *
_output_shapes
:*
	elem_type0
х
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/EnterEnterQtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
parallel_iterations 
щ
Wtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPushV2StackPushV2Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/Enterlstm_1/while/add_4^training/Adam/gradients/Add*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*'
_output_shapes
:џџџџџџџџџ*
swap_memory(
г
Vtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopV2
StackPopV2\training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopV2/Enter^training/Adam/gradients/Sub*
	elem_type0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*'
_output_shapes
:џџџџџџџџџ

\training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopV2/EnterEnterQtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum
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
StackPopV2htraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
_output_shapes
:*
	elem_type0
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
Htraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SelectSelectKtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqualAtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/ReshapeGtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros*'
_output_shapes
:џџџџџџџџџ*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum

Jtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Select_1SelectKtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqualGtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zerosAtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Reshape*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*'
_output_shapes
:џџџџџџџџџ*
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
:џџџџџџџџџ
ю
Gtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Sum_1SumJtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Select_1Ytraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs:1*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
:џџџџџџџџџ
Ю
Etraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/ShapeShapelstm_1/while/add_2*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
out_type0*
_output_shapes
:
п
Gtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_1Const^training/Adam/gradients/Sub*
_output_shapes
: *5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
valueB *
dtype0
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
Etraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/zerosFillGtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_2Ktraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros/Const*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*

index_type0*'
_output_shapes
:џџџџџџџџџ*
T0
ш
Itraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual	LessEqualTtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopV2Qtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/Const_1*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ
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
Otraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/EnterEnterOtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
parallel_iterations *
is_constant(
у
Utraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPushV2StackPushV2Otraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/Enterlstm_1/while/add_2^training/Adam/gradients/Add*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ*
swap_memory(
Э
Ttraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopV2
StackPopV2Ztraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopV2/Enter^training/Adam/gradients/Sub*'
_output_shapes
:џџџџџџџџџ*
	elem_type0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum
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
[training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
valueB :
џџџџџџџџџ
Й
[training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_accStackV2[training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum
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
atraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPushV2StackPushV2[training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/EnterEtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape^training/Adam/gradients/Add*
_output_shapes
:*
swap_memory(*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum
и
`training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ftraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum
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
Ftraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/SelectSelectItraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual?training/Adam/gradients/lstm_1/while/clip_by_value_grad/ReshapeEtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ

Htraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Select_1SelectItraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqualEtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros?training/Adam/gradients/lstm_1/while/clip_by_value_grad/Reshape*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ
р
Ctraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/SumSumFtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/SelectUtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum
№
Gtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/ReshapeReshapeCtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Sum`training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
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
7training/Adam/gradients/lstm_1/while/add_5_grad/Shape_1Shapelstm_1/while/MatMul_6*
_output_shapes
:*
T0*%
_class
loc:@lstm_1/while/add_5*
out_type0
ш
Etraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2Rtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/add_5*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *%
_class
loc:@lstm_1/while/add_5*
valueB :
џџџџџџџџџ

Ktraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/add_5*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_5
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
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/add_5
ъ
Vtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_5*
parallel_iterations *
is_constant(
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
Mtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_5
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
Xtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterMtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_5*
parallel_iterations 
Ё
3training/Adam/gradients/lstm_1/while/add_5_grad/SumSum7training/Adam/gradients/lstm_1/while/Tanh_grad/TanhGradEtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_5
А
7training/Adam/gradients/lstm_1/while/add_5_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_5_grad/SumPtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/add_5*
Tshape0*'
_output_shapes
:џџџџџџџџџ
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
:џџџџџџџџџ
ю
?training/Adam/gradients/lstm_1/while/BiasAdd_3_grad/BiasAddGradBiasAddGrad7training/Adam/gradients/lstm_1/while/add_7_grad/Reshape*
data_formatNHWC*
_output_shapes
:*
T0*)
_class
loc:@lstm_1/while/BiasAdd_3
С
9training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMulMatMul9training/Adam/gradients/lstm_1/while/add_7_grad/Reshape_1?training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul/Enter*
T0*(
_class
loc:@lstm_1/while/MatMul_7*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul/EnterEnterlstm_1/strided_slice_7*
is_constant(*
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_7*
parallel_iterations 
С
;training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV29training/Adam/gradients/lstm_1/while/add_7_grad/Reshape_1*
T0*(
_class
loc:@lstm_1/while/MatMul_7*
_output_shapes

:*
transpose_a(*
transpose_b( 
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
Atraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/f_accStackV2Atraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*F
_class<
:loc:@lstm_1/while/Identity_3loc:@lstm_1/while/MatMul_7
Ж
Atraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/EnterEnterAtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_7
П
Gtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPushV2StackPushV2Atraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/Enterlstm_1/while/Identity_3^training/Adam/gradients/Add*'
_output_shapes
:џџџџџџџџџ*
swap_memory(*
T0*(
_class
loc:@lstm_1/while/MatMul_7
Є
Ftraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV2
StackPopV2Ltraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV2/Enter^training/Adam/gradients/Sub*'
_output_shapes
:џџџџџџџџџ*
	elem_type0*(
_class
loc:@lstm_1/while/MatMul_7
й
Ltraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV2/EnterEnterAtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_7*
parallel_iterations 
Ў
5training/Adam/gradients/lstm_1/while/add_4_grad/ShapeShapelstm_1/while/mul_1*
_output_shapes
:*
T0*%
_class
loc:@lstm_1/while/add_4*
out_type0
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
Ktraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *%
_class
loc:@lstm_1/while/add_4*
valueB :
џџџџџџџџџ

Ktraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/add_4*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_4*
parallel_iterations 
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
:џџџџџџџџџ
З
5training/Adam/gradients/lstm_1/while/add_4_grad/Sum_1SumItraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/ReshapeGtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@lstm_1/while/add_4*
_output_shapes
:*
	keep_dims( *

Tidx0
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
Ktraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/add_2
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
Qtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/add_2_grad/Shape^training/Adam/gradients/Add*
_output_shapes
:*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/add_2
Ј
Ptraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/add_2
ъ
Vtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_2
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
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/add_2*
Tshape0
Е
5training/Adam/gradients/lstm_1/while/add_2_grad/Sum_1SumGtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/ReshapeGtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_2

9training/Adam/gradients/lstm_1/while/add_2_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_2_grad/Sum_17training/Adam/gradients/lstm_1/while/add_2_grad/Shape_1*
_output_shapes
: *
T0*%
_class
loc:@lstm_1/while/add_2*
Tshape0
ю
?training/Adam/gradients/lstm_1/while/BiasAdd_2_grad/BiasAddGradBiasAddGrad7training/Adam/gradients/lstm_1/while/add_5_grad/Reshape*
T0*)
_class
loc:@lstm_1/while/BiasAdd_2*
data_formatNHWC*
_output_shapes
:
С
9training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMulMatMul9training/Adam/gradients/lstm_1/while/add_5_grad/Reshape_1?training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMul/Enter*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0*(
_class
loc:@lstm_1/while/MatMul_6
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMul/EnterEnterlstm_1/strided_slice_6*
T0*(
_class
loc:@lstm_1/while/MatMul_6*
parallel_iterations *
is_constant(*
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
С
;training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV29training/Adam/gradients/lstm_1/while/add_5_grad/Reshape_1*
T0*(
_class
loc:@lstm_1/while/MatMul_6*
_output_shapes

:*
transpose_a(*
transpose_b( 
П
9training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMulMatMul7training/Adam/gradients/lstm_1/while/add_7_grad/Reshape?training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul/Enter*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0*(
_class
loc:@lstm_1/while/MatMul_3
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul/EnterEnterlstm_1/strided_slice_3*
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_3*
parallel_iterations *
is_constant(
П
;training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV27training/Adam/gradients/lstm_1/while/add_7_grad/Reshape*(
_class
loc:@lstm_1/while/MatMul_3*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
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
Atraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_accStackV2Atraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/Const*M
_classC
Aloc:@lstm_1/while/MatMul_3#loc:@lstm_1/while/TensorArrayReadV3*

stack_name *
_output_shapes
:*
	elem_type0
Ж
Atraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/EnterEnterAtraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_acc*(
_class
loc:@lstm_1/while/MatMul_3*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0
Ц
Gtraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPushV2StackPushV2Atraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/Enterlstm_1/while/TensorArrayReadV3^training/Adam/gradients/Add*
T0*(
_class
loc:@lstm_1/while/MatMul_3*'
_output_shapes
:џџџџџџџџџ*
swap_memory(
Є
Ftraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV2
StackPopV2Ltraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV2/Enter^training/Adam/gradients/Sub*(
_class
loc:@lstm_1/while/MatMul_3*'
_output_shapes
:џџџџџџџџџ*
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
valueB*    *
dtype0*
_output_shapes
:
г
Atraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_1Enter?training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter*
parallel_iterations *
is_constant( *
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
З
Atraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_2MergeAtraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_1Gtraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/NextIteration*
N*
_output_shapes

:: *
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter

@training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/SwitchSwitchAtraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter* 
_output_shapes
::

=training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/AddAddBtraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/Switch:1?training/Adam/gradients/lstm_1/while/BiasAdd_3_grad/BiasAddGrad*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter*
_output_shapes
:
э
Gtraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/NextIterationNextIteration=training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/Add*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter*
_output_shapes
:*
T0
с
Atraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_3Exit@training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/Switch*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter*
_output_shapes
:*
T0
У
>training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_accConst*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*
valueB*    *
dtype0*
_output_shapes

:
д
@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc*
T0*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*
parallel_iterations *
is_constant( *
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
З
@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/NextIteration*
T0*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*
N* 
_output_shapes
:: 

?training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*(
_output_shapes
::

<training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1*
T0*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*
_output_shapes

:
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/Add*
_output_shapes

:*
T0*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter
т
@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/Switch*
_output_shapes

:*
T0*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter
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
Etraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/lstm_1/while/mul_1_grad/ShapePtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/ConstConst*
_output_shapes
: *%
_class
loc:@lstm_1/while/mul_1*
valueB :
џџџџџџџџџ*
dtype0
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
StackPopV2Vtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/mul_1
ъ
Vtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_1

3training/Adam/gradients/lstm_1/while/mul_1_grad/MulMul7training/Adam/gradients/lstm_1/while/add_4_grad/Reshape>training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_1*'
_output_shapes
:џџџџџџџџџ
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
:џџџџџџџџџ*
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
:џџџџџџџџџ*
	elem_type0
Ц
Dtraining/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPopV2/EnterEnter9training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_1*
parallel_iterations 
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
7training/Adam/gradients/lstm_1/while/mul_1_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/mul_1_grad/Sum5training/Adam/gradients/lstm_1/while/mul_1_grad/Shape*
_output_shapes
: *
T0*%
_class
loc:@lstm_1/while/mul_1*
Tshape0

5training/Adam/gradients/lstm_1/while/mul_1_grad/Mul_1Mul;training/Adam/gradients/lstm_1/while/mul_1_grad/Mul_1/Const7training/Adam/gradients/lstm_1/while/add_4_grad/Reshape*%
_class
loc:@lstm_1/while/mul_1*'
_output_shapes
:џџџџџџџџџ*
T0
Х
;training/Adam/gradients/lstm_1/while/mul_1_grad/Mul_1/ConstConst^training/Adam/gradients/Sub*
_output_shapes
: *%
_class
loc:@lstm_1/while/mul_1*
valueB
 *ЭЬL>*
dtype0
Ѓ
5training/Adam/gradients/lstm_1/while/mul_1_grad/Sum_1Sum5training/Adam/gradients/lstm_1/while/mul_1_grad/Mul_1Gtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@lstm_1/while/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
Д
9training/Adam/gradients/lstm_1/while/mul_1_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/mul_1_grad/Sum_1Ptraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_1*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Й
3training/Adam/gradients/lstm_1/while/mul_grad/ShapeConst^training/Adam/gradients/Sub*
_output_shapes
: *#
_class
loc:@lstm_1/while/mul*
valueB *
dtype0
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
Itraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/ConstConst*
_output_shapes
: *#
_class
loc:@lstm_1/while/mul*
valueB :
џџџџџџџџџ*
dtype0

Itraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_accStackV2Itraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*#
_class
loc:@lstm_1/while/mul
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
Otraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Itraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/mul_grad/Shape_1^training/Adam/gradients/Add*#
_class
loc:@lstm_1/while/mul*
_output_shapes
:*
swap_memory(*
T0
Ђ
Ntraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ttraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
	elem_type0*#
_class
loc:@lstm_1/while/mul*
_output_shapes
:
ф
Ttraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterItraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_acc*
T0*#
_class
loc:@lstm_1/while/mul*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context

1training/Adam/gradients/lstm_1/while/mul_grad/MulMul7training/Adam/gradients/lstm_1/while/add_2_grad/Reshape<training/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*#
_class
loc:@lstm_1/while/mul
Р
7training/Adam/gradients/lstm_1/while/mul_grad/Mul/ConstConst*
_output_shapes
: *<
_class2
0loc:@lstm_1/while/add_1loc:@lstm_1/while/mul*
valueB :
џџџџџџџџџ*
dtype0
ј
7training/Adam/gradients/lstm_1/while/mul_grad/Mul/f_accStackV27training/Adam/gradients/lstm_1/while/mul_grad/Mul/Const*<
_class2
0loc:@lstm_1/while/add_1loc:@lstm_1/while/mul*

stack_name *
_output_shapes
:*
	elem_type0

7training/Adam/gradients/lstm_1/while/mul_grad/Mul/EnterEnter7training/Adam/gradients/lstm_1/while/mul_grad/Mul/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*#
_class
loc:@lstm_1/while/mul*
parallel_iterations 
Ё
=training/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPushV2StackPushV27training/Adam/gradients/lstm_1/while/mul_grad/Mul/Enterlstm_1/while/add_1^training/Adam/gradients/Add*
T0*#
_class
loc:@lstm_1/while/mul*'
_output_shapes
:џџџџџџџџџ*
swap_memory(

<training/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPopV2
StackPopV2Btraining/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*#
_class
loc:@lstm_1/while/mul*'
_output_shapes
:џџџџџџџџџ*
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
1training/Adam/gradients/lstm_1/while/mul_grad/SumSum1training/Adam/gradients/lstm_1/while/mul_grad/MulCtraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*#
_class
loc:@lstm_1/while/mul
ќ
5training/Adam/gradients/lstm_1/while/mul_grad/ReshapeReshape1training/Adam/gradients/lstm_1/while/mul_grad/Sum3training/Adam/gradients/lstm_1/while/mul_grad/Shape*
_output_shapes
: *
T0*#
_class
loc:@lstm_1/while/mul*
Tshape0

3training/Adam/gradients/lstm_1/while/mul_grad/Mul_1Mul9training/Adam/gradients/lstm_1/while/mul_grad/Mul_1/Const7training/Adam/gradients/lstm_1/while/add_2_grad/Reshape*
T0*#
_class
loc:@lstm_1/while/mul*'
_output_shapes
:џџџџџџџџџ
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
3training/Adam/gradients/lstm_1/while/mul_grad/Sum_1Sum3training/Adam/gradients/lstm_1/while/mul_grad/Mul_1Etraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*#
_class
loc:@lstm_1/while/mul*
_output_shapes
:
Ќ
7training/Adam/gradients/lstm_1/while/mul_grad/Reshape_1Reshape3training/Adam/gradients/lstm_1/while/mul_grad/Sum_1Ntraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*#
_class
loc:@lstm_1/while/mul*
Tshape0*'
_output_shapes
:џџџџџџџџџ
П
9training/Adam/gradients/lstm_1/while/MatMul_2_grad/MatMulMatMul7training/Adam/gradients/lstm_1/while/add_5_grad/Reshape?training/Adam/gradients/lstm_1/while/MatMul_2_grad/MatMul/Enter*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0*(
_class
loc:@lstm_1/while/MatMul_2
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_2_grad/MatMul/EnterEnterlstm_1/strided_slice_2*
is_constant(*
_output_shapes

:*B

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

:*
transpose_a(*
transpose_b( 
Н
?training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_accConst*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter*
valueB*    *
dtype0*
_output_shapes
:
г
Atraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc_1Enter?training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc*
is_constant( *
_output_shapes
:*B

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

:: 

@training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/SwitchSwitchAtraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter* 
_output_shapes
::

=training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/AddAddBtraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/Switch:1?training/Adam/gradients/lstm_1/while/BiasAdd_2_grad/BiasAddGrad*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter*
_output_shapes
:
э
Gtraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/NextIterationNextIteration=training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/Add*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter*
_output_shapes
:
с
Atraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc_3Exit@training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/Switch*
_output_shapes
:*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter
У
>training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_accConst*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*
valueB*    *
dtype0*
_output_shapes

:
д
@training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc*
T0*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*
parallel_iterations *
is_constant( *
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
З
@training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/NextIteration*
T0*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*
N* 
_output_shapes
:: 

?training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*(
_output_shapes
::

<training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMul_1*
T0*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*
_output_shapes

:
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/Add*
_output_shapes

:*
T0*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter
т
@training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/Switch*
T0*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*
_output_shapes

:
У
>training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_accConst*
dtype0*
_output_shapes

:*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
valueB*    
д
@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc*
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
parallel_iterations *
is_constant( *
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
З
@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/NextIteration*
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
N* 
_output_shapes
:: 

?training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*(
_output_shapes
::

<training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1*
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
_output_shapes

:
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/Add*
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
_output_shapes

:
т
@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/Switch*
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
_output_shapes

:
А
:training/Adam/gradients/lstm_1/strided_slice_11_grad/ShapeConst**
_class 
loc:@lstm_1/strided_slice_11*
valueB: *
dtype0*
_output_shapes
:
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
: 
Е
9training/Adam/gradients/lstm_1/strided_slice_7_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_7*
valueB"       *
dtype0*
_output_shapes
:
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_7_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_7_grad/Shapelstm_1/strided_slice_7/stacklstm_1/strided_slice_7/stack_1lstm_1/strided_slice_7/stack_2@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_3*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

: *
T0*
Index0*)
_class
loc:@lstm_1/strided_slice_7
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
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
	elem_type0*%
_class
loc:@lstm_1/while/add_3*
_output_shapes
:
ъ
Vtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_3
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
Straining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/add_3_grad/Shape_1^training/Adam/gradients/Add*
_output_shapes
:*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/add_3
Ќ
Rtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_3*
_output_shapes
:*
	elem_type0
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
3training/Adam/gradients/lstm_1/while/add_3_grad/SumSum9training/Adam/gradients/lstm_1/while/mul_1_grad/Reshape_1Etraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_3
А
7training/Adam/gradients/lstm_1/while/add_3_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_3_grad/SumPtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/add_3*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ї
5training/Adam/gradients/lstm_1/while/add_3_grad/Sum_1Sum9training/Adam/gradients/lstm_1/while/mul_1_grad/Reshape_1Gtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_3*
_output_shapes
:
Ж
9training/Adam/gradients/lstm_1/while/add_3_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_3_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2_1*'
_output_shapes
:џџџџџџџџџ*
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
7training/Adam/gradients/lstm_1/while/add_1_grad/Shape_1Shapelstm_1/while/MatMul_4*
T0*%
_class
loc:@lstm_1/while/add_1*
out_type0*
_output_shapes
:
ш
Etraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2Rtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/add_1
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
Qtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/add_1_grad/Shape^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/add_1*
_output_shapes
:*
swap_memory(
Ј
Ptraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_1*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc*%
_class
loc:@lstm_1/while/add_1*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0
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
Mtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Const_1*%
_class
loc:@lstm_1/while/add_1*

stack_name *
_output_shapes
:*
	elem_type0
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
Straining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/add_1_grad/Shape_1^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/add_1*
_output_shapes
:*
swap_memory(
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
3training/Adam/gradients/lstm_1/while/add_1_grad/SumSum7training/Adam/gradients/lstm_1/while/mul_grad/Reshape_1Etraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_1*
_output_shapes
:
А
7training/Adam/gradients/lstm_1/while/add_1_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_1_grad/SumPtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
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
9training/Adam/gradients/lstm_1/while/add_1_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_1_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2_1*%
_class
loc:@lstm_1/while/add_1*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
У
>training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_accConst*
_output_shapes

:*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
valueB*    *
dtype0
д
@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc*
is_constant( *
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
parallel_iterations 
З
@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/NextIteration*
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
N* 
_output_shapes
:: 

?training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*(
_output_shapes
::

<training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_2_grad/MatMul_1*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
_output_shapes

:*
T0
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/Add*
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
_output_shapes

:
т
@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/Switch*
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
_output_shapes

:
А
:training/Adam/gradients/lstm_1/strided_slice_10_grad/ShapeConst**
_class 
loc:@lstm_1/strided_slice_10*
valueB: *
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
: *
T0*
Index0**
_class 
loc:@lstm_1/strided_slice_10
Е
9training/Adam/gradients/lstm_1/strided_slice_6_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_6*
valueB"       *
dtype0*
_output_shapes
:
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_6_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_6_grad/Shapelstm_1/strided_slice_6/stacklstm_1/strided_slice_6/stack_1lstm_1/strided_slice_6/stack_2@training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc_3*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

: *
T0*
Index0*)
_class
loc:@lstm_1/strided_slice_6
Е
9training/Adam/gradients/lstm_1/strided_slice_3_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_3*
valueB"       *
dtype0*
_output_shapes
:
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_3_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_3_grad/Shapelstm_1/strided_slice_3/stacklstm_1/strided_slice_3/stack_1lstm_1/strided_slice_3/stack_2@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_3*
Index0*
T0*)
_class
loc:@lstm_1/strided_slice_3*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

: 
ю
?training/Adam/gradients/lstm_1/while/BiasAdd_1_grad/BiasAddGradBiasAddGrad7training/Adam/gradients/lstm_1/while/add_3_grad/Reshape*
data_formatNHWC*
_output_shapes
:*
T0*)
_class
loc:@lstm_1/while/BiasAdd_1
С
9training/Adam/gradients/lstm_1/while/MatMul_5_grad/MatMulMatMul9training/Adam/gradients/lstm_1/while/add_3_grad/Reshape_1?training/Adam/gradients/lstm_1/while/MatMul_5_grad/MatMul/Enter*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0*(
_class
loc:@lstm_1/while/MatMul_5
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_5_grad/MatMul/EnterEnterlstm_1/strided_slice_5*
is_constant(*
_output_shapes

:*B

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

:*
transpose_a(*
transpose_b( 
ъ
=training/Adam/gradients/lstm_1/while/BiasAdd_grad/BiasAddGradBiasAddGrad7training/Adam/gradients/lstm_1/while/add_1_grad/Reshape*
T0*'
_class
loc:@lstm_1/while/BiasAdd*
data_formatNHWC*
_output_shapes
:
С
9training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMulMatMul9training/Adam/gradients/lstm_1/while/add_1_grad/Reshape_1?training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMul/Enter*
T0*(
_class
loc:@lstm_1/while/MatMul_4*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMul/EnterEnterlstm_1/strided_slice_4*
T0*(
_class
loc:@lstm_1/while/MatMul_4*
parallel_iterations *
is_constant(*
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
С
;training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV29training/Adam/gradients/lstm_1/while/add_1_grad/Reshape_1*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0*(
_class
loc:@lstm_1/while/MatMul_4
Е
9training/Adam/gradients/lstm_1/strided_slice_2_grad/ShapeConst*
dtype0*
_output_shapes
:*)
_class
loc:@lstm_1/strided_slice_2*
valueB"       
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_2_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_2_grad/Shapelstm_1/strided_slice_2/stacklstm_1/strided_slice_2/stack_1lstm_1/strided_slice_2/stack_2@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_3*
T0*
Index0*)
_class
loc:@lstm_1/strided_slice_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

: 
П
9training/Adam/gradients/lstm_1/while/MatMul_1_grad/MatMulMatMul7training/Adam/gradients/lstm_1/while/add_3_grad/Reshape?training/Adam/gradients/lstm_1/while/MatMul_1_grad/MatMul/Enter*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0*(
_class
loc:@lstm_1/while/MatMul_1
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_1_grad/MatMul/EnterEnterlstm_1/strided_slice_1*
parallel_iterations *
is_constant(*
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_1
П
;training/Adam/gradients/lstm_1/while/MatMul_1_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV27training/Adam/gradients/lstm_1/while/add_3_grad/Reshape*(
_class
loc:@lstm_1/while/MatMul_1*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
Н
?training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_accConst*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter*
valueB*    *
dtype0*
_output_shapes
:
г
Atraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_1Enter?training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc*
parallel_iterations *
is_constant( *
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter
З
Atraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_2MergeAtraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_1Gtraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/NextIteration*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter*
N*
_output_shapes

:: *
T0

@training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/SwitchSwitchAtraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2* 
_output_shapes
::*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter

=training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/AddAddBtraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/Switch:1?training/Adam/gradients/lstm_1/while/BiasAdd_1_grad/BiasAddGrad*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter*
_output_shapes
:*
T0
э
Gtraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/NextIterationNextIteration=training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/Add*
_output_shapes
:*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter
с
Atraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_3Exit@training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/Switch*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter*
_output_shapes
:
У
>training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_accConst*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter*
valueB*    *
dtype0*
_output_shapes

:
д
@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc*
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter*
parallel_iterations *
is_constant( 
З
@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/NextIteration*
T0*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter*
N* 
_output_shapes
:: 

?training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter*(
_output_shapes
::*
T0

<training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_5_grad/MatMul_1*
T0*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter*
_output_shapes

:
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/Add*
T0*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter*
_output_shapes

:
т
@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/Switch*
_output_shapes

:*
T0*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter
Й
7training/Adam/gradients/lstm_1/while/MatMul_grad/MatMulMatMul7training/Adam/gradients/lstm_1/while/add_1_grad/Reshape=training/Adam/gradients/lstm_1/while/MatMul_grad/MatMul/Enter*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0*&
_class
loc:@lstm_1/while/MatMul

=training/Adam/gradients/lstm_1/while/MatMul_grad/MatMul/EnterEnterlstm_1/strided_slice*
is_constant(*
_output_shapes

:*B

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

:*
transpose_a(
Й
=training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_accConst*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter*
valueB*    *
dtype0*
_output_shapes
:
Э
?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_1Enter=training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc*
is_constant( *
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter*
parallel_iterations 
Џ
?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_2Merge?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_1Etraining/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/NextIteration*
N*
_output_shapes

:: *
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter

>training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/SwitchSwitch?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter* 
_output_shapes
::

;training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/AddAdd@training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/Switch:1=training/Adam/gradients/lstm_1/while/BiasAdd_grad/BiasAddGrad*
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter*
_output_shapes
:
ч
Etraining/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/NextIterationNextIteration;training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/Add*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter*
_output_shapes
:*
T0
л
?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_3Exit>training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/Switch*
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter*
_output_shapes
:
ї
training/Adam/gradients/AddN_5AddN9training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul9training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMul9training/Adam/gradients/lstm_1/while/MatMul_5_grad/MatMul9training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMul*
T0*(
_class
loc:@lstm_1/while/MatMul_7*
N*'
_output_shapes
:џџџџџџџџџ
У
>training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_accConst*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter*
valueB*    *
dtype0*
_output_shapes

:
д
@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc*
parallel_iterations *
is_constant( *
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter
З
@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/NextIteration*
T0*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter*
N* 
_output_shapes
:: 

?training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter*(
_output_shapes
::*
T0

<training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMul_1*
T0*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter*
_output_shapes

:
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/Add*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter*
_output_shapes

:*
T0
т
@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter*
_output_shapes

:*
T0
У
>training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_accConst*
dtype0*
_output_shapes

:*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
valueB*    
д
@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc*
is_constant( *
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
parallel_iterations 
З
@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/NextIteration*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
N* 
_output_shapes
:: *
T0

?training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*(
_output_shapes
::*
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter

<training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_1_grad/MatMul_1*
_output_shapes

:*
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/Add*
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
_output_shapes

:
т
@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/Switch*
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
_output_shapes

:
Ў
9training/Adam/gradients/lstm_1/strided_slice_9_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_9*
valueB: *
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
: 
Е
9training/Adam/gradients/lstm_1/strided_slice_5_grad/ShapeConst*
dtype0*
_output_shapes
:*)
_class
loc:@lstm_1/strided_slice_5*
valueB"       
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_5_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_5_grad/Shapelstm_1/strided_slice_5/stacklstm_1/strided_slice_5/stack_1lstm_1/strided_slice_5/stack_2@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_3*
Index0*
T0*)
_class
loc:@lstm_1/strided_slice_5*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

: 
П
<training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_accConst*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
valueB*    *
dtype0*
_output_shapes

:
Ю
>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_1Enter<training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc*
is_constant( *
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
parallel_iterations 
Џ
>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_2Merge>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_1Dtraining/Adam/gradients/lstm_1/while/MatMul/Enter_grad/NextIteration*
N* 
_output_shapes
:: *
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter

=training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/SwitchSwitch>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*(
_output_shapes
::*
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter

:training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/AddAdd?training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/Switch:19training/Adam/gradients/lstm_1/while/MatMul_grad/MatMul_1*
_output_shapes

:*
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter
ш
Dtraining/Adam/gradients/lstm_1/while/MatMul/Enter_grad/NextIterationNextIteration:training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/Add*
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
_output_shapes

:
м
>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_3Exit=training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/Switch*
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
_output_shapes

:
Ў
9training/Adam/gradients/lstm_1/strided_slice_8_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_8*
valueB: *
dtype0*
_output_shapes
:
ѓ
Dtraining/Adam/gradients/lstm_1/strided_slice_8_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_8_grad/Shapelstm_1/strided_slice_8/stacklstm_1/strided_slice_8/stack_1lstm_1/strided_slice_8/stack_2?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_3*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*)
_class
loc:@lstm_1/strided_slice_8*
shrink_axis_mask 
Ю
Btraining/Adam/gradients/lstm_1/while/Switch_3_grad_1/NextIterationNextIterationtraining/Adam/gradients/AddN_5*'
_class
loc:@lstm_1/while/Merge_3*'
_output_shapes
:џџџџџџџџџ*
T0
Е
9training/Adam/gradients/lstm_1/strided_slice_4_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_4*
valueB"       *
dtype0*
_output_shapes
:
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_4_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_4_grad/Shapelstm_1/strided_slice_4/stacklstm_1/strided_slice_4/stack_1lstm_1/strided_slice_4/stack_2@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_3*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

: *
Index0*
T0*)
_class
loc:@lstm_1/strided_slice_4
Е
9training/Adam/gradients/lstm_1/strided_slice_1_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_1*
valueB"       *
dtype0*
_output_shapes
:
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_1_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_1_grad/Shapelstm_1/strided_slice_1/stacklstm_1/strided_slice_1/stack_1lstm_1/strided_slice_1/stack_2@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_3*
end_mask*
_output_shapes

: *
Index0*
T0*)
_class
loc:@lstm_1/strided_slice_1*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask 
Б
7training/Adam/gradients/lstm_1/strided_slice_grad/ShapeConst*'
_class
loc:@lstm_1/strided_slice*
valueB"       *
dtype0*
_output_shapes
:
ъ
Btraining/Adam/gradients/lstm_1/strided_slice_grad/StridedSliceGradStridedSliceGrad7training/Adam/gradients/lstm_1/strided_slice_grad/Shapelstm_1/strided_slice/stacklstm_1/strided_slice/stack_1lstm_1/strided_slice/stack_2>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_3*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

: *
T0*
Index0*'
_class
loc:@lstm_1/strided_slice*
shrink_axis_mask 

training/Adam/gradients/AddN_6AddNEtraining/Adam/gradients/lstm_1/strided_slice_11_grad/StridedSliceGradEtraining/Adam/gradients/lstm_1/strided_slice_10_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_9_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_8_grad/StridedSliceGrad*
T0**
_class 
loc:@lstm_1/strided_slice_11*
N*
_output_shapes
: 

training/Adam/gradients/AddN_7AddNDtraining/Adam/gradients/lstm_1/strided_slice_7_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_6_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_5_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_4_grad/StridedSliceGrad*
N*
_output_shapes

: *
T0*)
_class
loc:@lstm_1/strided_slice_7

training/Adam/gradients/AddN_8AddNDtraining/Adam/gradients/lstm_1/strided_slice_3_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_2_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_1_grad/StridedSliceGradBtraining/Adam/gradients/lstm_1/strided_slice_grad/StridedSliceGrad*
_output_shapes

: *
T0*)
_class
loc:@lstm_1/strided_slice_3*
N
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
training/Adam/CastCastAdam/iterations/read*
_output_shapes
: *

DstT0*

SrcT0	*
Truncate( 
X
training/Adam/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
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
training/Adam/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
T0*
_output_shapes
: 
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
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_1*
_output_shapes
: *
T0
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
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
T0*
_output_shapes
: 
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
T0*
_output_shapes
: 
^
training/Adam/mulMulAdam/lr/readtraining/Adam/truediv*
_output_shapes
: *
T0
h
training/Adam/zerosConst*
valueB *    *
dtype0*
_output_shapes

: 

training/Adam/Variable
VariableV2*
dtype0*
_output_shapes

: *
	container *
shape
: *
shared_name 
б
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*
validate_shape(*
_output_shapes

: *
use_locking(*
T0*)
_class
loc:@training/Adam/Variable

training/Adam/Variable/readIdentitytraining/Adam/Variable*
T0*)
_class
loc:@training/Adam/Variable*
_output_shapes

: 
j
training/Adam/zeros_1Const*
valueB *    *
dtype0*
_output_shapes

: 

training/Adam/Variable_1
VariableV2*
_output_shapes

: *
	container *
shape
: *
shared_name *
dtype0
й
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes

: 

training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
T0*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes

: 
b
training/Adam/zeros_2Const*
valueB *    *
dtype0*
_output_shapes
: 

training/Adam/Variable_2
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
е
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*
_output_shapes
: 

training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*
T0*+
_class!
loc:@training/Adam/Variable_2*
_output_shapes
: 
v
%training/Adam/zeros_3/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"      
`
training/Adam/zeros_3/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_3Fill%training/Adam/zeros_3/shape_as_tensortraining/Adam/zeros_3/Const*
T0*

index_type0* 
_output_shapes
:


training/Adam/Variable_3
VariableV2*
shape:
*
shared_name *
dtype0* 
_output_shapes
:
*
	container 
л
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/zeros_3*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3* 
_output_shapes
:
*
T0*+
_class!
loc:@training/Adam/Variable_3
d
training/Adam/zeros_4Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_4
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
ж
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/zeros_4*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*
T0*+
_class!
loc:@training/Adam/Variable_4*
_output_shapes	
:
v
%training/Adam/zeros_5/shape_as_tensorConst*
valueB"      *
dtype0*
_output_shapes
:
`
training/Adam/zeros_5/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_5Fill%training/Adam/zeros_5/shape_as_tensortraining/Adam/zeros_5/Const*
T0*

index_type0* 
_output_shapes
:


training/Adam/Variable_5
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
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(* 
_output_shapes
:


training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5* 
_output_shapes
:
*
T0*+
_class!
loc:@training/Adam/Variable_5
d
training/Adam/zeros_6Const*
dtype0*
_output_shapes	
:*
valueB*    

training/Adam/Variable_6
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
ж
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*
_output_shapes	
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(

training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
T0*+
_class!
loc:@training/Adam/Variable_6*
_output_shapes	
:
l
training/Adam/zeros_7Const*
valueB	*    *
dtype0*
_output_shapes
:	

training/Adam/Variable_7
VariableV2*
shared_name *
dtype0*
_output_shapes
:	*
	container *
shape:	
к
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/zeros_7*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7

training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*
T0*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes
:	
b
training/Adam/zeros_8Const*
dtype0*
_output_shapes
:*
valueB*    

training/Adam/Variable_8
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
е
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(

training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*
T0*+
_class!
loc:@training/Adam/Variable_8*
_output_shapes
:
j
training/Adam/zeros_9Const*
valueB *    *
dtype0*
_output_shapes

: 

training/Adam/Variable_9
VariableV2*
_output_shapes

: *
	container *
shape
: *
shared_name *
dtype0
й
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*
_output_shapes

: *
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(

training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
T0*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes

: 
k
training/Adam/zeros_10Const*
valueB *    *
dtype0*
_output_shapes

: 

training/Adam/Variable_10
VariableV2*
dtype0*
_output_shapes

: *
	container *
shape
: *
shared_name 
н
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes

: *
use_locking(*
T0

training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*
T0*,
_class"
 loc:@training/Adam/Variable_10*
_output_shapes

: 
c
training/Adam/zeros_11Const*
valueB *    *
dtype0*
_output_shapes
: 

training/Adam/Variable_11
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
й
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/zeros_11*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
: 

training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
: *
T0
w
&training/Adam/zeros_12/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"      
a
training/Adam/zeros_12/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Ё
training/Adam/zeros_12Fill&training/Adam/zeros_12/shape_as_tensortraining/Adam/zeros_12/Const* 
_output_shapes
:
*
T0*

index_type0

training/Adam/Variable_12
VariableV2*
shared_name *
dtype0* 
_output_shapes
:
*
	container *
shape:

п
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
T0*,
_class"
 loc:@training/Adam/Variable_12* 
_output_shapes
:

e
training/Adam/zeros_13Const*
dtype0*
_output_shapes	
:*
valueB*    

training/Adam/Variable_13
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
к
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*
_output_shapes	
:*
T0*,
_class"
 loc:@training/Adam/Variable_13
w
&training/Adam/zeros_14/shape_as_tensorConst*
_output_shapes
:*
valueB"      *
dtype0
a
training/Adam/zeros_14/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ё
training/Adam/zeros_14Fill&training/Adam/zeros_14/shape_as_tensortraining/Adam/zeros_14/Const*
T0*

index_type0* 
_output_shapes
:


training/Adam/Variable_14
VariableV2* 
_output_shapes
:
*
	container *
shape:
*
shared_name *
dtype0
п
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(* 
_output_shapes
:
*
use_locking(

training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
T0*,
_class"
 loc:@training/Adam/Variable_14* 
_output_shapes
:

e
training/Adam/zeros_15Const*
_output_shapes	
:*
valueB*    *
dtype0

training/Adam/Variable_15
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
к
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/zeros_15*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes	
:*
use_locking(

training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*
T0*,
_class"
 loc:@training/Adam/Variable_15*
_output_shapes	
:
m
training/Adam/zeros_16Const*
valueB	*    *
dtype0*
_output_shapes
:	

training/Adam/Variable_16
VariableV2*
shared_name *
dtype0*
_output_shapes
:	*
	container *
shape:	
о
 training/Adam/Variable_16/AssignAssigntraining/Adam/Variable_16training/Adam/zeros_16*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(*
_output_shapes
:	

training/Adam/Variable_16/readIdentitytraining/Adam/Variable_16*
T0*,
_class"
 loc:@training/Adam/Variable_16*
_output_shapes
:	
c
training/Adam/zeros_17Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_17
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
й
 training/Adam/Variable_17/AssignAssigntraining/Adam/Variable_17training/Adam/zeros_17*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_17/readIdentitytraining/Adam/Variable_17*
T0*,
_class"
 loc:@training/Adam/Variable_17*
_output_shapes
:
p
&training/Adam/zeros_18/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_18/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_18Fill&training/Adam/zeros_18/shape_as_tensortraining/Adam/zeros_18/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_18
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
й
 training/Adam/Variable_18/AssignAssigntraining/Adam/Variable_18training/Adam/zeros_18*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(

training/Adam/Variable_18/readIdentitytraining/Adam/Variable_18*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_18
p
&training/Adam/zeros_19/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_19/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_19Fill&training/Adam/zeros_19/shape_as_tensortraining/Adam/zeros_19/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_19
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
й
 training/Adam/Variable_19/AssignAssigntraining/Adam/Variable_19training/Adam/zeros_19*
T0*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_19/readIdentitytraining/Adam/Variable_19*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_19
p
&training/Adam/zeros_20/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_20/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_20Fill&training/Adam/zeros_20/shape_as_tensortraining/Adam/zeros_20/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_20
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
й
 training/Adam/Variable_20/AssignAssigntraining/Adam/Variable_20training/Adam/zeros_20*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20

training/Adam/Variable_20/readIdentitytraining/Adam/Variable_20*
T0*,
_class"
 loc:@training/Adam/Variable_20*
_output_shapes
:
p
&training/Adam/zeros_21/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_21/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_21Fill&training/Adam/zeros_21/shape_as_tensortraining/Adam/zeros_21/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_21
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
й
 training/Adam/Variable_21/AssignAssigntraining/Adam/Variable_21training/Adam/zeros_21*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_21

training/Adam/Variable_21/readIdentitytraining/Adam/Variable_21*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_21
p
&training/Adam/zeros_22/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_22/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_22Fill&training/Adam/zeros_22/shape_as_tensortraining/Adam/zeros_22/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_22
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
й
 training/Adam/Variable_22/AssignAssigntraining/Adam/Variable_22training/Adam/zeros_22*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_22/readIdentitytraining/Adam/Variable_22*,
_class"
 loc:@training/Adam/Variable_22*
_output_shapes
:*
T0
p
&training/Adam/zeros_23/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_23/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_23Fill&training/Adam/zeros_23/shape_as_tensortraining/Adam/zeros_23/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_23
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
й
 training/Adam/Variable_23/AssignAssigntraining/Adam/Variable_23training/Adam/zeros_23*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_23*
validate_shape(*
_output_shapes
:

training/Adam/Variable_23/readIdentitytraining/Adam/Variable_23*
T0*,
_class"
 loc:@training/Adam/Variable_23*
_output_shapes
:
p
&training/Adam/zeros_24/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_24/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_24Fill&training/Adam/zeros_24/shape_as_tensortraining/Adam/zeros_24/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_24
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
й
 training/Adam/Variable_24/AssignAssigntraining/Adam/Variable_24training/Adam/zeros_24*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_24*
validate_shape(*
_output_shapes
:

training/Adam/Variable_24/readIdentitytraining/Adam/Variable_24*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_24
p
&training/Adam/zeros_25/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_25/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_25Fill&training/Adam/zeros_25/shape_as_tensortraining/Adam/zeros_25/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_25
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
й
 training/Adam/Variable_25/AssignAssigntraining/Adam/Variable_25training/Adam/zeros_25*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_25*
validate_shape(*
_output_shapes
:

training/Adam/Variable_25/readIdentitytraining/Adam/Variable_25*
T0*,
_class"
 loc:@training/Adam/Variable_25*
_output_shapes
:
p
&training/Adam/zeros_26/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_26/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_26Fill&training/Adam/zeros_26/shape_as_tensortraining/Adam/zeros_26/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_26
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
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
training/Adam/Variable_26/readIdentitytraining/Adam/Variable_26*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_26
r
training/Adam/mul_1MulAdam/beta_1/readtraining/Adam/Variable/read*
_output_shapes

: *
T0
Z
training/Adam/sub_2/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_2Subtraining/Adam/sub_2/xAdam/beta_1/read*
T0*
_output_shapes
: 
x
training/Adam/mul_2Multraining/Adam/sub_2training/Adam/gradients/AddN_8*
T0*
_output_shapes

: 
m
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*
_output_shapes

: *
T0
t
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_9/read*
_output_shapes

: *
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
g
training/Adam/SquareSquaretraining/Adam/gradients/AddN_8*
T0*
_output_shapes

: 
n
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
_output_shapes

: *
T0
m
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
_output_shapes

: *
T0
k
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
T0*
_output_shapes

: 
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

%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_3*
_output_shapes

: *
T0

training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_2*
T0*
_output_shapes

: 
d
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0*
_output_shapes

: 
Z
training/Adam/add_3/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
p
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*
T0*
_output_shapes

: 
u
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
_output_shapes

: *
T0
p
training/Adam/sub_4Sublstm_1/kernel/readtraining/Adam/truediv_1*
T0*
_output_shapes

: 
Ш
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*
_output_shapes

: *
use_locking(
Ю
training/Adam/Assign_1Assigntraining/Adam/Variable_9training/Adam/add_2*
validate_shape(*
_output_shapes

: *
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9
И
training/Adam/Assign_2Assignlstm_1/kerneltraining/Adam/sub_4*
_output_shapes

: *
use_locking(*
T0* 
_class
loc:@lstm_1/kernel*
validate_shape(
t
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
T0*
_output_shapes

: 
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
x
training/Adam/mul_7Multraining/Adam/sub_5training/Adam/gradients/AddN_7*
T0*
_output_shapes

: 
m
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
T0*
_output_shapes

: 
u
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_10/read*
T0*
_output_shapes

: 
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
i
training/Adam/Square_1Squaretraining/Adam/gradients/AddN_7*
T0*
_output_shapes

: 
p
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
T0*
_output_shapes

: 
m
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
T0*
_output_shapes

: 
l
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes

: 
Z
training/Adam/Const_4Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_5Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_5*
T0*
_output_shapes

: 

training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_4*
_output_shapes

: *
T0
d
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
T0*
_output_shapes

: 
Z
training/Adam/add_6/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
p
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
T0*
_output_shapes

: 
v
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
_output_shapes

: *
T0
z
training/Adam/sub_7Sublstm_1/recurrent_kernel/readtraining/Adam/truediv_2*
T0*
_output_shapes

: 
Ю
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes

: 
а
training/Adam/Assign_4Assigntraining/Adam/Variable_10training/Adam/add_5*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes

: 
Ь
training/Adam/Assign_5Assignlstm_1/recurrent_kerneltraining/Adam/sub_7*
validate_shape(*
_output_shapes

: *
use_locking(*
T0**
_class 
loc:@lstm_1/recurrent_kernel
q
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*
_output_shapes
: *
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
u
training/Adam/mul_12Multraining/Adam/sub_8training/Adam/gradients/AddN_6*
_output_shapes
: *
T0
k
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
T0*
_output_shapes
: 
r
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_11/read*
T0*
_output_shapes
: 
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
e
training/Adam/Square_2Squaretraining/Adam/gradients/AddN_6*
T0*
_output_shapes
: 
m
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*
_output_shapes
: 
k
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
_output_shapes
: *
T0
h
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*
_output_shapes
: *
T0
Z
training/Adam/Const_6Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_7Const*
dtype0*
_output_shapes
: *
valueB
 *  

%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_7*
T0*
_output_shapes
: 

training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_6*
T0*
_output_shapes
: 
`
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0*
_output_shapes
: 
Z
training/Adam/add_9/yConst*
_output_shapes
: *
valueB
 *Пж3*
dtype0
l
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y*
_output_shapes
: *
T0
r
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
T0*
_output_shapes
: 
k
training/Adam/sub_10Sublstm_1/bias/readtraining/Adam/truediv_3*
_output_shapes
: *
T0
Ъ
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*
_output_shapes
: 
Ь
training/Adam/Assign_7Assigntraining/Adam/Variable_11training/Adam/add_8*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
: 
Б
training/Adam/Assign_8Assignlstm_1/biastraining/Adam/sub_10*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@lstm_1/bias*
validate_shape(
w
training/Adam/mul_16MulAdam/beta_1/readtraining/Adam/Variable_3/read* 
_output_shapes
:
*
T0
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

training/Adam/mul_17Multraining/Adam/sub_114training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

r
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17* 
_output_shapes
:
*
T0
x
training/Adam/mul_18MulAdam/beta_2/readtraining/Adam/Variable_12/read*
T0* 
_output_shapes
:

[
training/Adam/sub_12/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_12Subtraining/Adam/sub_12/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_3Square4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

t
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3* 
_output_shapes
:
*
T0
r
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
T0* 
_output_shapes
:

o
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
T0* 
_output_shapes
:

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

%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_9*
T0* 
_output_shapes
:


training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_8*
T0* 
_output_shapes
:

f
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
T0* 
_output_shapes
:

[
training/Adam/add_12/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
t
training/Adam/add_12Addtraining/Adam/Sqrt_4training/Adam/add_12/y* 
_output_shapes
:
*
T0
y
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12* 
_output_shapes
:
*
T0
t
training/Adam/sub_13Subdense_1/kernel/readtraining/Adam/truediv_4*
T0* 
_output_shapes
:

б
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(* 
_output_shapes
:

д
training/Adam/Assign_10Assigntraining/Adam/Variable_12training/Adam/add_11*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(* 
_output_shapes
:
*
use_locking(
О
training/Adam/Assign_11Assigndense_1/kerneltraining/Adam/sub_13*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
r
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*
_output_shapes	
:*
T0
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

training/Adam/mul_22Multraining/Adam/sub_148training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
m
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
T0*
_output_shapes	
:
s
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_13/read*
T0*
_output_shapes	
:
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

training/Adam/Square_4Square8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
o
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
T0*
_output_shapes	
:
m
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
_output_shapes	
:*
T0
j
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
T0*
_output_shapes	
:
[
training/Adam/Const_10Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_11Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_11*
_output_shapes	
:*
T0

training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_10*
_output_shapes	
:*
T0
a
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
_output_shapes	
:*
T0
[
training/Adam/add_15/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
o
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y*
_output_shapes	
:*
T0
t
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
_output_shapes	
:*
T0
m
training/Adam/sub_16Subdense_1/bias/readtraining/Adam/truediv_5*
T0*
_output_shapes	
:
Э
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4
Я
training/Adam/Assign_13Assigntraining/Adam/Variable_13training/Adam/add_14*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(
Е
training/Adam/Assign_14Assigndense_1/biastraining/Adam/sub_16*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(
w
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read* 
_output_shapes
:
*
T0
[
training/Adam/sub_17/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_17Subtraining/Adam/sub_17/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_27Multraining/Adam/sub_174training/Adam/gradients/dense_2/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
r
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
T0* 
_output_shapes
:

x
training/Adam/mul_28MulAdam/beta_2/readtraining/Adam/Variable_14/read*
T0* 
_output_shapes
:

[
training/Adam/sub_18/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_18Subtraining/Adam/sub_18/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_5Square4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

t
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
T0* 
_output_shapes
:

r
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
T0* 
_output_shapes
:

o
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16* 
_output_shapes
:
*
T0
[
training/Adam/Const_12Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_13Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_13*
T0* 
_output_shapes
:


training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_12*
T0* 
_output_shapes
:

f
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
T0* 
_output_shapes
:

[
training/Adam/add_18/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
t
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y* 
_output_shapes
:
*
T0
y
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
T0* 
_output_shapes
:

t
training/Adam/sub_19Subdense_2/kernel/readtraining/Adam/truediv_6*
T0* 
_output_shapes
:

в
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(* 
_output_shapes
:
*
use_locking(
д
training/Adam/Assign_16Assigntraining/Adam/Variable_14training/Adam/add_17*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(* 
_output_shapes
:

О
training/Adam/Assign_17Assigndense_2/kerneltraining/Adam/sub_19*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(* 
_output_shapes
:

r
training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*
_output_shapes	
:*
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

training/Adam/mul_32Multraining/Adam/sub_208training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
m
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
_output_shapes	
:*
T0
s
training/Adam/mul_33MulAdam/beta_2/readtraining/Adam/Variable_15/read*
_output_shapes	
:*
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

training/Adam/Square_6Square8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
o
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
_output_shapes	
:*
T0
m
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
T0*
_output_shapes	
:
j
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
T0*
_output_shapes	
:
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

%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_15*
_output_shapes	
:*
T0

training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_14*
_output_shapes	
:*
T0
a
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
_output_shapes	
:*
T0
[
training/Adam/add_21/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
o
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*
_output_shapes	
:*
T0
t
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
T0*
_output_shapes	
:
m
training/Adam/sub_22Subdense_2/bias/readtraining/Adam/truediv_7*
_output_shapes	
:*
T0
Э
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Я
training/Adam/Assign_19Assigntraining/Adam/Variable_15training/Adam/add_20*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes	
:
Е
training/Adam/Assign_20Assigndense_2/biastraining/Adam/sub_22*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes	
:
v
training/Adam/mul_36MulAdam/beta_1/readtraining/Adam/Variable_7/read*
T0*
_output_shapes
:	
[
training/Adam/sub_23/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_23Subtraining/Adam/sub_23/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_37Multraining/Adam/sub_234training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	
q
training/Adam/add_22Addtraining/Adam/mul_36training/Adam/mul_37*
T0*
_output_shapes
:	
w
training/Adam/mul_38MulAdam/beta_2/readtraining/Adam/Variable_16/read*
T0*
_output_shapes
:	
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

training/Adam/Square_7Square4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	
s
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7*
_output_shapes
:	*
T0
q
training/Adam/add_23Addtraining/Adam/mul_38training/Adam/mul_39*
T0*
_output_shapes
:	
n
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22*
T0*
_output_shapes
:	
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

%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_17*
_output_shapes
:	*
T0

training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_16*
T0*
_output_shapes
:	
e
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
T0*
_output_shapes
:	
[
training/Adam/add_24/yConst*
_output_shapes
: *
valueB
 *Пж3*
dtype0
s
training/Adam/add_24Addtraining/Adam/Sqrt_8training/Adam/add_24/y*
T0*
_output_shapes
:	
x
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24*
T0*
_output_shapes
:	
s
training/Adam/sub_25Subdense_3/kernel/readtraining/Adam/truediv_8*
_output_shapes
:	*
T0
б
training/Adam/Assign_21Assigntraining/Adam/Variable_7training/Adam/add_22*
_output_shapes
:	*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(
г
training/Adam/Assign_22Assigntraining/Adam/Variable_16training/Adam/add_23*
_output_shapes
:	*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(
Н
training/Adam/Assign_23Assigndense_3/kerneltraining/Adam/sub_25*
_output_shapes
:	*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(
q
training/Adam/mul_41MulAdam/beta_1/readtraining/Adam/Variable_8/read*
T0*
_output_shapes
:
[
training/Adam/sub_26/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_26Subtraining/Adam/sub_26/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_42Multraining/Adam/sub_268training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
l
training/Adam/add_25Addtraining/Adam/mul_41training/Adam/mul_42*
T0*
_output_shapes
:
r
training/Adam/mul_43MulAdam/beta_2/readtraining/Adam/Variable_17/read*
_output_shapes
:*
T0
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

training/Adam/Square_8Square8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
n
training/Adam/mul_44Multraining/Adam/sub_27training/Adam/Square_8*
_output_shapes
:*
T0
l
training/Adam/add_26Addtraining/Adam/mul_43training/Adam/mul_44*
T0*
_output_shapes
:
i
training/Adam/mul_45Multraining/Adam/multraining/Adam/add_25*
_output_shapes
:*
T0
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

%training/Adam/clip_by_value_9/MinimumMinimumtraining/Adam/add_26training/Adam/Const_19*
T0*
_output_shapes
:

training/Adam/clip_by_value_9Maximum%training/Adam/clip_by_value_9/Minimumtraining/Adam/Const_18*
_output_shapes
:*
T0
`
training/Adam/Sqrt_9Sqrttraining/Adam/clip_by_value_9*
_output_shapes
:*
T0
[
training/Adam/add_27/yConst*
dtype0*
_output_shapes
: *
valueB
 *Пж3
n
training/Adam/add_27Addtraining/Adam/Sqrt_9training/Adam/add_27/y*
_output_shapes
:*
T0
s
training/Adam/truediv_9RealDivtraining/Adam/mul_45training/Adam/add_27*
T0*
_output_shapes
:
l
training/Adam/sub_28Subdense_3/bias/readtraining/Adam/truediv_9*
_output_shapes
:*
T0
Ь
training/Adam/Assign_24Assigntraining/Adam/Variable_8training/Adam/add_25*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*
_output_shapes
:
Ю
training/Adam/Assign_25Assigntraining/Adam/Variable_17training/Adam/add_26*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17
Д
training/Adam/Assign_26Assigndense_3/biastraining/Adam/sub_28*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:

training/group_depsNoOp	^loss/mul^metrics/acc/Mean^training/Adam/Assign^training/Adam/AssignAdd^training/Adam/Assign_1^training/Adam/Assign_10^training/Adam/Assign_11^training/Adam/Assign_12^training/Adam/Assign_13^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_18^training/Adam/Assign_19^training/Adam/Assign_2^training/Adam/Assign_20^training/Adam/Assign_21^training/Adam/Assign_22^training/Adam/Assign_23^training/Adam/Assign_24^training/Adam/Assign_25^training/Adam/Assign_26^training/Adam/Assign_3^training/Adam/Assign_4^training/Adam/Assign_5^training/Adam/Assign_6^training/Adam/Assign_7^training/Adam/Assign_8^training/Adam/Assign_9

IsVariableInitializedIsVariableInitializedlstm_1/kernel* 
_class
loc:@lstm_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_1IsVariableInitializedlstm_1/recurrent_kernel*
dtype0*
_output_shapes
: **
_class 
loc:@lstm_1/recurrent_kernel

IsVariableInitialized_2IsVariableInitializedlstm_1/bias*
_class
loc:@lstm_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_3IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_4IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_5IsVariableInitializeddense_2/kernel*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_6IsVariableInitializeddense_2/bias*
_class
loc:@dense_2/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_7IsVariableInitializeddense_3/kernel*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_8IsVariableInitializeddense_3/bias*
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_9IsVariableInitializedAdam/iterations*"
_class
loc:@Adam/iterations*
dtype0	*
_output_shapes
: 
{
IsVariableInitialized_10IsVariableInitializedAdam/lr*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 

IsVariableInitialized_11IsVariableInitializedAdam/beta_1*
_output_shapes
: *
_class
loc:@Adam/beta_1*
dtype0

IsVariableInitialized_12IsVariableInitializedAdam/beta_2*
_output_shapes
: *
_class
loc:@Adam/beta_2*
dtype0

IsVariableInitialized_13IsVariableInitialized
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 

IsVariableInitialized_14IsVariableInitializedtraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
dtype0*
_output_shapes
: 

IsVariableInitialized_15IsVariableInitializedtraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_16IsVariableInitializedtraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0*
_output_shapes
: 

IsVariableInitialized_17IsVariableInitializedtraining/Adam/Variable_3*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_3

IsVariableInitialized_18IsVariableInitializedtraining/Adam/Variable_4*+
_class!
loc:@training/Adam/Variable_4*
dtype0*
_output_shapes
: 

IsVariableInitialized_19IsVariableInitializedtraining/Adam/Variable_5*+
_class!
loc:@training/Adam/Variable_5*
dtype0*
_output_shapes
: 

IsVariableInitialized_20IsVariableInitializedtraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*
dtype0*
_output_shapes
: 

IsVariableInitialized_21IsVariableInitializedtraining/Adam/Variable_7*+
_class!
loc:@training/Adam/Variable_7*
dtype0*
_output_shapes
: 

IsVariableInitialized_22IsVariableInitializedtraining/Adam/Variable_8*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_8

IsVariableInitialized_23IsVariableInitializedtraining/Adam/Variable_9*+
_class!
loc:@training/Adam/Variable_9*
dtype0*
_output_shapes
: 

IsVariableInitialized_24IsVariableInitializedtraining/Adam/Variable_10*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_10*
dtype0

IsVariableInitialized_25IsVariableInitializedtraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
dtype0*
_output_shapes
: 

IsVariableInitialized_26IsVariableInitializedtraining/Adam/Variable_12*,
_class"
 loc:@training/Adam/Variable_12*
dtype0*
_output_shapes
: 

IsVariableInitialized_27IsVariableInitializedtraining/Adam/Variable_13*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_13

IsVariableInitialized_28IsVariableInitializedtraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
dtype0*
_output_shapes
: 

IsVariableInitialized_29IsVariableInitializedtraining/Adam/Variable_15*,
_class"
 loc:@training/Adam/Variable_15*
dtype0*
_output_shapes
: 

IsVariableInitialized_30IsVariableInitializedtraining/Adam/Variable_16*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_16

IsVariableInitialized_31IsVariableInitializedtraining/Adam/Variable_17*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_17

IsVariableInitialized_32IsVariableInitializedtraining/Adam/Variable_18*,
_class"
 loc:@training/Adam/Variable_18*
dtype0*
_output_shapes
: 

IsVariableInitialized_33IsVariableInitializedtraining/Adam/Variable_19*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_19*
dtype0

IsVariableInitialized_34IsVariableInitializedtraining/Adam/Variable_20*,
_class"
 loc:@training/Adam/Variable_20*
dtype0*
_output_shapes
: 

IsVariableInitialized_35IsVariableInitializedtraining/Adam/Variable_21*,
_class"
 loc:@training/Adam/Variable_21*
dtype0*
_output_shapes
: 

IsVariableInitialized_36IsVariableInitializedtraining/Adam/Variable_22*,
_class"
 loc:@training/Adam/Variable_22*
dtype0*
_output_shapes
: 

IsVariableInitialized_37IsVariableInitializedtraining/Adam/Variable_23*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_23

IsVariableInitialized_38IsVariableInitializedtraining/Adam/Variable_24*,
_class"
 loc:@training/Adam/Variable_24*
dtype0*
_output_shapes
: 

IsVariableInitialized_39IsVariableInitializedtraining/Adam/Variable_25*,
_class"
 loc:@training/Adam/Variable_25*
dtype0*
_output_shapes
: 

IsVariableInitialized_40IsVariableInitializedtraining/Adam/Variable_26*,
_class"
 loc:@training/Adam/Variable_26*
dtype0*
_output_shapes
: 
№	
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^dense_3/bias/Assign^dense_3/kernel/Assign^lstm_1/bias/Assign^lstm_1/kernel/Assign^lstm_1/recurrent_kernel/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign!^training/Adam/Variable_18/Assign!^training/Adam/Variable_19/Assign ^training/Adam/Variable_2/Assign!^training/Adam/Variable_20/Assign!^training/Adam/Variable_21/Assign!^training/Adam/Variable_22/Assign!^training/Adam/Variable_23/Assign!^training/Adam/Variable_24/Assign!^training/Adam/Variable_25/Assign!^training/Adam/Variable_26/Assign ^training/Adam/Variable_3/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign"wМЛЦМ     яўq	'§ЭћжAJЏ
4ј3
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
Ttype*1.12.02v1.12.0-rc2-3-ga6d8ffae09цЙ
y
lstm_1_inputPlaceholder*
dtype0*,
_output_shapes
:џџџџџџџџџ*!
shape:џџџџџџџџџ
l
lstm_1/random_uniform/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
^
lstm_1/random_uniform/minConst*
valueB
 *JQкО*
dtype0*
_output_shapes
: 
^
lstm_1/random_uniform/maxConst*
valueB
 *JQк>*
dtype0*
_output_shapes
: 
І
#lstm_1/random_uniform/RandomUniformRandomUniformlstm_1/random_uniform/shape*
dtype0*
_output_shapes

: *
seed2ќ*
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

: *
T0
{
lstm_1/random_uniformAddlstm_1/random_uniform/mullstm_1/random_uniform/min*
T0*
_output_shapes

: 

lstm_1/kernel
VariableV2*
dtype0*
_output_shapes

: *
	container *
shape
: *
shared_name 
И
lstm_1/kernel/AssignAssignlstm_1/kernellstm_1/random_uniform*
use_locking(*
T0* 
_class
loc:@lstm_1/kernel*
validate_shape(*
_output_shapes

: 
x
lstm_1/kernel/readIdentitylstm_1/kernel*
T0* 
_class
loc:@lstm_1/kernel*
_output_shapes

: 
њ
%lstm_1/recurrent_kernel/initial_valueConst*
valueB "№Gб=qЯ	Н$MЂММ7>ЄxоННІНC=,>f#шН3y=^jzН_ОЪ5>SRЉ=-яIОкЗ>Пa6ОГк=іНРvОб@НYв=Hс?<Eх>HВОNК>
­=кuОљ>N3M>"\Y>І/Н <OмМНЫчИН!JDНPсЪ>жK=^ЛїЏНкWНќ%+ОQ>M>м2 =ЉMЉНZ>Г#О§њ$ОUдМQ>иW= а>ёE>=ОО}M>JсМ'ННr>р[ыН c9<чr>пъООе-Д=№>YЙ=К!ОфICОИкМmОGПНИОћНъ)mН@7 МUЧН/-GИLЄ=ђyОЪNЋ<u>ОЮУЇ> LОHНЁМ_ИxО[3НѕuОiБМНW=&EF>щRЩ=Пyт=l&ѓ>Т=Ўo>є;>уйНЛ:>TuЌНПср>>+(	>Щ>^Нci(ОиMSМ~їYОљa=ОКсN>cЎЧМcО}ЋОtR>Г=@џ$О^дЬН}ОкФОхaIО\ъЊНCLОЄЬНSWО-<>ЌОX{ЙМM==РкђНёn=+WМ	Њ6ОTОo'ОыОAТНYJ=лу!<б8>&МX[>)YМхї}>)оО1>у1>ўЦг=чgМuЇ=>8$Оb2О,тщ=	ЂЎ=йа:=лxA<#>IЯб>ЦDОВU=ї n;у.ОИиЇМp@Оy=ёџ<ъНиЉОТYFНш(Є>щ:НБ1Ч=ГеЊМѕC>.мъ= Ў<yКОOОФ(=О_b-ОњЮЖМuD:Н&Cг=хџA>§!ОЩЊ=ОїэО,еОАWЛ0ЅОјЇКђЯcО>$=CЕ=P>ЙF=ъ<5> ЃО	рЭ>">ЭjНbФтНШмНЊНj/ОсЯ\>Ъ>=њYОWэпНѕЙ<О9И>k*=]'Нlр=CНХЧНБv7Н эуМфЯ+>pшu>pо=>лZП>[і= ЛВНдО1>	Оy­=|=>жЖ<[ИНbн
>В
=AУ;О/жНљР= L=>В,<МTОЫ<мОЛњ)>&ЌН$Ор ОЏ6>8zНNН4>ыЊEНВДK>F}ОЛНкQОoСLО*
dtype0*
_output_shapes

: 

lstm_1/recurrent_kernel
VariableV2*
shared_name *
dtype0*
_output_shapes

: *
	container *
shape
: 
ц
lstm_1/recurrent_kernel/AssignAssignlstm_1/recurrent_kernel%lstm_1/recurrent_kernel/initial_value*
_output_shapes

: *
use_locking(*
T0**
_class 
loc:@lstm_1/recurrent_kernel*
validate_shape(

lstm_1/recurrent_kernel/readIdentitylstm_1/recurrent_kernel*
T0**
_class 
loc:@lstm_1/recurrent_kernel*
_output_shapes

: 
Y
lstm_1/ConstConst*
_output_shapes
:*
valueB*    *
dtype0
[
lstm_1/Const_1Const*
dtype0*
_output_shapes
:*
valueB*  ?
[
lstm_1/Const_2Const*
_output_shapes
:*
valueB*    *
dtype0
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
: *

Tidx0*
T0
w
lstm_1/bias
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
І
lstm_1/bias/AssignAssignlstm_1/biaslstm_1/concat*
use_locking(*
T0*
_class
loc:@lstm_1/bias*
validate_shape(*
_output_shapes
: 
n
lstm_1/bias/readIdentitylstm_1/bias*
T0*
_class
loc:@lstm_1/bias*
_output_shapes
: 
k
lstm_1/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
m
lstm_1/strided_slice/stack_1Const*
valueB"       *
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
lstm_1/strided_sliceStridedSlicelstm_1/kernel/readlstm_1/strided_slice/stacklstm_1/strided_slice/stack_1lstm_1/strided_slice/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:*
T0*
Index0*
shrink_axis_mask 
m
lstm_1/strided_slice_1/stackConst*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_1/stack_1Const*
valueB"       *
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
lstm_1/strided_slice_1StridedSlicelstm_1/kernel/readlstm_1/strided_slice_1/stacklstm_1/strided_slice_1/stack_1lstm_1/strided_slice_1/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:*
T0*
Index0
m
lstm_1/strided_slice_2/stackConst*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
o
lstm_1/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
В
lstm_1/strided_slice_2StridedSlicelstm_1/kernel/readlstm_1/strided_slice_2/stacklstm_1/strided_slice_2/stack_1lstm_1/strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:
m
lstm_1/strided_slice_3/stackConst*
valueB"       *
dtype0*
_output_shapes
:
o
lstm_1/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        
o
lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
В
lstm_1/strided_slice_3StridedSlicelstm_1/kernel/readlstm_1/strided_slice_3/stacklstm_1/strided_slice_3/stack_1lstm_1/strided_slice_3/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:*
Index0*
T0
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
valueB"       *
dtype0
o
lstm_1/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
М
lstm_1/strided_slice_4StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_4/stacklstm_1/strided_slice_4/stack_1lstm_1/strided_slice_4/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:*
T0*
Index0*
shrink_axis_mask 
m
lstm_1/strided_slice_5/stackConst*
dtype0*
_output_shapes
:*
valueB"       
o
lstm_1/strided_slice_5/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
o
lstm_1/strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
М
lstm_1/strided_slice_5StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_5/stacklstm_1/strided_slice_5/stack_1lstm_1/strided_slice_5/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:*
T0*
Index0
m
lstm_1/strided_slice_6/stackConst*
_output_shapes
:*
valueB"       *
dtype0
o
lstm_1/strided_slice_6/stack_1Const*
valueB"       *
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
lstm_1/strided_slice_6StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_6/stacklstm_1/strided_slice_6/stack_1lstm_1/strided_slice_6/stack_2*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:*
Index0*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_7/stackConst*
valueB"       *
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
lstm_1/strided_slice_7/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
М
lstm_1/strided_slice_7StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_7/stacklstm_1/strided_slice_7/stack_1lstm_1/strided_slice_7/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:*
T0*
Index0
f
lstm_1/strided_slice_8/stackConst*
dtype0*
_output_shapes
:*
valueB: 
h
lstm_1/strided_slice_8/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
h
lstm_1/strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ќ
lstm_1/strided_slice_8StridedSlicelstm_1/bias/readlstm_1/strided_slice_8/stacklstm_1/strided_slice_8/stack_1lstm_1/strided_slice_8/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:
f
lstm_1/strided_slice_9/stackConst*
_output_shapes
:*
valueB:*
dtype0
h
lstm_1/strided_slice_9/stack_1Const*
valueB:*
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
lstm_1/strided_slice_9StridedSlicelstm_1/bias/readlstm_1/strided_slice_9/stacklstm_1/strided_slice_9/stack_1lstm_1/strided_slice_9/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
:
g
lstm_1/strided_slice_10/stackConst*
valueB:*
dtype0*
_output_shapes
:
i
lstm_1/strided_slice_10/stack_1Const*
valueB:*
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
lstm_1/strided_slice_10StridedSlicelstm_1/bias/readlstm_1/strided_slice_10/stacklstm_1/strided_slice_10/stack_1lstm_1/strided_slice_10/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0
g
lstm_1/strided_slice_11/stackConst*
valueB:*
dtype0*
_output_shapes
:
i
lstm_1/strided_slice_11/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
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
:*
T0*
Index0
c
lstm_1/zeros_like	ZerosLikelstm_1_input*,
_output_shapes
:џџџџџџџџџ*
T0
m
lstm_1/Sum/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:


lstm_1/SumSumlstm_1/zeros_likelstm_1/Sum/reduction_indices*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( *

Tidx0*
T0
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
dtype0*
_output_shapes
:*
valueB"      

lstm_1/TileTilelstm_1/ExpandDimslstm_1/Tile/multiples*
T0*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0
h
lstm_1/Tile_1/multiplesConst*
valueB"      *
dtype0*
_output_shapes
:

lstm_1/Tile_1Tilelstm_1/ExpandDimslstm_1/Tile_1/multiples*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
j
lstm_1/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:

lstm_1/transpose	Transposelstm_1_inputlstm_1/transpose/perm*
T0*,
_output_shapes
:џџџџџџџџџ*
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
lstm_1/strided_slice_12/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
i
lstm_1/strided_slice_12/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ј
lstm_1/strided_slice_12StridedSlicelstm_1/Shapelstm_1/strided_slice_12/stacklstm_1/strided_slice_12/stack_1lstm_1/strided_slice_12/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
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
lstm_1/strided_slice_13StridedSlicelstm_1/transposelstm_1/strided_slice_13/stacklstm_1/strided_slice_13/stack_1lstm_1/strided_slice_13/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMulMatMullstm_1/strided_slice_13lstm_1/strided_slice*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
Ђ
lstm_1/MatMul_1MatMullstm_1/strided_slice_13lstm_1/strided_slice_1*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
Ђ
lstm_1/MatMul_2MatMullstm_1/strided_slice_13lstm_1/strided_slice_2*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
Ђ
lstm_1/MatMul_3MatMullstm_1/strided_slice_13lstm_1/strided_slice_3*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

lstm_1/BiasAddBiasAddlstm_1/MatMullstm_1/strided_slice_8*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0

lstm_1/BiasAdd_1BiasAddlstm_1/MatMul_1lstm_1/strided_slice_9*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0

lstm_1/BiasAdd_2BiasAddlstm_1/MatMul_2lstm_1/strided_slice_10*'
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC

lstm_1/BiasAdd_3BiasAddlstm_1/MatMul_3lstm_1/strided_slice_11*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMul_4MatMullstm_1/Tilelstm_1/strided_slice_4*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
d

lstm_1/addAddlstm_1/BiasAddlstm_1/MatMul_4*
T0*'
_output_shapes
:џџџџџџџџџ
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
:џџџџџџџџџ
S
lstm_1/add_1/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
a
lstm_1/add_1Add
lstm_1/mullstm_1/add_1/y*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/Const_3Const*
dtype0*
_output_shapes
: *
valueB
 *    
S
lstm_1/Const_4Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
w
lstm_1/clip_by_value/MinimumMinimumlstm_1/add_1lstm_1/Const_4*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/clip_by_valueMaximumlstm_1/clip_by_value/Minimumlstm_1/Const_3*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMul_5MatMullstm_1/Tilelstm_1/strided_slice_5*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
h
lstm_1/add_2Addlstm_1/BiasAdd_1lstm_1/MatMul_5*
T0*'
_output_shapes
:џџџџџџџџџ
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
:џџџџџџџџџ*
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
:џџџџџџџџџ
S
lstm_1/Const_5Const*
_output_shapes
: *
valueB
 *    *
dtype0
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
:џџџџџџџџџ*
T0

lstm_1/clip_by_value_1Maximumlstm_1/clip_by_value_1/Minimumlstm_1/Const_5*
T0*'
_output_shapes
:џџџџџџџџџ
l
lstm_1/mul_2Mullstm_1/clip_by_value_1lstm_1/Tile_1*'
_output_shapes
:џџџџџџџџџ*
T0

lstm_1/MatMul_6MatMullstm_1/Tilelstm_1/strided_slice_6*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
h
lstm_1/add_4Addlstm_1/BiasAdd_2lstm_1/MatMul_6*'
_output_shapes
:џџџџџџџџџ*
T0
S
lstm_1/TanhTanhlstm_1/add_4*'
_output_shapes
:џџџџџџџџџ*
T0
h
lstm_1/mul_3Mullstm_1/clip_by_valuelstm_1/Tanh*'
_output_shapes
:џџџџџџџџџ*
T0
a
lstm_1/add_5Addlstm_1/mul_2lstm_1/mul_3*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMul_7MatMullstm_1/Tilelstm_1/strided_slice_7*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 
h
lstm_1/add_6Addlstm_1/BiasAdd_3lstm_1/MatMul_7*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/mul_4/xConst*
_output_shapes
: *
valueB
 *ЭЬL>*
dtype0
c
lstm_1/mul_4Mullstm_1/mul_4/xlstm_1/add_6*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/add_7/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
c
lstm_1/add_7Addlstm_1/mul_4lstm_1/add_7/y*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/Const_7Const*
valueB
 *    *
dtype0*
_output_shapes
: 
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
:џџџџџџџџџ

lstm_1/clip_by_value_2Maximumlstm_1/clip_by_value_2/Minimumlstm_1/Const_7*'
_output_shapes
:џџџџџџџџџ*
T0
U
lstm_1/Tanh_1Tanhlstm_1/add_5*'
_output_shapes
:џџџџџџџџџ*
T0
l
lstm_1/mul_5Mullstm_1/clip_by_value_2lstm_1/Tanh_1*
T0*'
_output_shapes
:џџџџџџџџџ
ь
lstm_1/TensorArrayTensorArrayV3lstm_1/strided_slice_12* 
tensor_array_name	output_ta*
dtype0*
_output_shapes

:: *
element_shape:*
clear_after_read(*
dynamic_size( *
identical_element_shapes(
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
lstm_1/TensorArrayUnstack/ShapeShapelstm_1/transpose*
_output_shapes
:*
T0*
out_type0
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
'lstm_1/TensorArrayUnstack/strided_sliceStridedSlicelstm_1/TensorArrayUnstack/Shape-lstm_1/TensorArrayUnstack/strided_slice/stack/lstm_1/TensorArrayUnstack/strided_slice/stack_1/lstm_1/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
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
lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
value
B :*
dtype0
`
lstm_1/while/iteration_counterConst*
dtype0*
_output_shapes
: *
value	B : 
Ж
lstm_1/while/EnterEnterlstm_1/while/iteration_counter*
_output_shapes
: **

frame_namelstm_1/while/while_context*
T0*
is_constant( *
parallel_iterations 
Ѕ
lstm_1/while/Enter_1Enterlstm_1/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: **

frame_namelstm_1/while/while_context
Ў
lstm_1/while/Enter_2Enterlstm_1/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: **

frame_namelstm_1/while/while_context
Ж
lstm_1/while/Enter_3Enterlstm_1/Tile*
T0*
is_constant( *
parallel_iterations *'
_output_shapes
:џџџџџџџџџ**

frame_namelstm_1/while/while_context
И
lstm_1/while/Enter_4Enterlstm_1/Tile_1*
T0*
is_constant( *
parallel_iterations *'
_output_shapes
:џџџџџџџџџ**

frame_namelstm_1/while/while_context
w
lstm_1/while/MergeMergelstm_1/while/Enterlstm_1/while/NextIteration*
_output_shapes
: : *
T0*
N
}
lstm_1/while/Merge_1Mergelstm_1/while/Enter_1lstm_1/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
}
lstm_1/while/Merge_2Mergelstm_1/while/Enter_2lstm_1/while/NextIteration_2*
_output_shapes
: : *
T0*
N

lstm_1/while/Merge_3Mergelstm_1/while/Enter_3lstm_1/while/NextIteration_3*
N*)
_output_shapes
:џџџџџџџџџ: *
T0

lstm_1/while/Merge_4Mergelstm_1/while/Enter_4lstm_1/while/NextIteration_4*
T0*
N*)
_output_shapes
:џџџџџџџџџ: 
g
lstm_1/while/LessLesslstm_1/while/Mergelstm_1/while/Less/Enter*
_output_shapes
: *
T0
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
lstm_1/while/SwitchSwitchlstm_1/while/Mergelstm_1/while/LoopCond*%
_class
loc:@lstm_1/while/Merge*
_output_shapes
: : *
T0
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
lstm_1/while/Switch_3Switchlstm_1/while/Merge_3lstm_1/while/LoopCond*
T0*'
_class
loc:@lstm_1/while/Merge_3*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
К
lstm_1/while/Switch_4Switchlstm_1/while/Merge_4lstm_1/while/LoopCond*
T0*'
_class
loc:@lstm_1/while/Merge_4*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
Y
lstm_1/while/IdentityIdentitylstm_1/while/Switch:1*
_output_shapes
: *
T0
]
lstm_1/while/Identity_1Identitylstm_1/while/Switch_1:1*
T0*
_output_shapes
: 
]
lstm_1/while/Identity_2Identitylstm_1/while/Switch_2:1*
_output_shapes
: *
T0
n
lstm_1/while/Identity_3Identitylstm_1/while/Switch_3:1*'
_output_shapes
:џџџџџџџџџ*
T0
n
lstm_1/while/Identity_4Identitylstm_1/while/Switch_4:1*'
_output_shapes
:џџџџџџџџџ*
T0
l
lstm_1/while/add/yConst^lstm_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
c
lstm_1/while/addAddlstm_1/while/Identitylstm_1/while/add/y*
T0*
_output_shapes
: 
а
lstm_1/while/TensorArrayReadV3TensorArrayReadV3$lstm_1/while/TensorArrayReadV3/Enterlstm_1/while/Identity_1&lstm_1/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:џџџџџџџџџ
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
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Л
lstm_1/while/MatMul/EnterEnterlstm_1/strided_slice*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context
Д
lstm_1/while/MatMul_1MatMullstm_1/while/TensorArrayReadV3lstm_1/while/MatMul_1/Enter*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
П
lstm_1/while/MatMul_1/EnterEnterlstm_1/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context
Д
lstm_1/while/MatMul_2MatMullstm_1/while/TensorArrayReadV3lstm_1/while/MatMul_2/Enter*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
П
lstm_1/while/MatMul_2/EnterEnterlstm_1/strided_slice_2*
is_constant(*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context*
T0
Д
lstm_1/while/MatMul_3MatMullstm_1/while/TensorArrayReadV3lstm_1/while/MatMul_3/Enter*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
П
lstm_1/while/MatMul_3/EnterEnterlstm_1/strided_slice_3*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context

lstm_1/while/BiasAddBiasAddlstm_1/while/MatMullstm_1/while/BiasAdd/Enter*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
К
lstm_1/while/BiasAdd/EnterEnterlstm_1/strided_slice_8*
parallel_iterations *
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*
is_constant(

lstm_1/while/BiasAdd_1BiasAddlstm_1/while/MatMul_1lstm_1/while/BiasAdd_1/Enter*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
М
lstm_1/while/BiasAdd_1/EnterEnterlstm_1/strided_slice_9*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelstm_1/while/while_context

lstm_1/while/BiasAdd_2BiasAddlstm_1/while/MatMul_2lstm_1/while/BiasAdd_2/Enter*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
Н
lstm_1/while/BiasAdd_2/EnterEnterlstm_1/strided_slice_10*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelstm_1/while/while_context

lstm_1/while/BiasAdd_3BiasAddlstm_1/while/MatMul_3lstm_1/while/BiasAdd_3/Enter*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
Н
lstm_1/while/BiasAdd_3/EnterEnterlstm_1/strided_slice_11*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelstm_1/while/while_context
­
lstm_1/while/MatMul_4MatMullstm_1/while/Identity_3lstm_1/while/MatMul_4/Enter*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
П
lstm_1/while/MatMul_4/EnterEnterlstm_1/strided_slice_4*
is_constant(*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context*
T0
x
lstm_1/while/add_1Addlstm_1/while/BiasAddlstm_1/while/MatMul_4*'
_output_shapes
:џџџџџџџџџ*
T0
o
lstm_1/while/mul/xConst^lstm_1/while/Identity*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
q
lstm_1/while/mulMullstm_1/while/mul/xlstm_1/while/add_1*'
_output_shapes
:џџџџџџџџџ*
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
:џџџџџџџџџ
o
lstm_1/while/ConstConst^lstm_1/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *    
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
:џџџџџџџџџ

lstm_1/while/clip_by_valueMaximum"lstm_1/while/clip_by_value/Minimumlstm_1/while/Const*
T0*'
_output_shapes
:џџџџџџџџџ
­
lstm_1/while/MatMul_5MatMullstm_1/while/Identity_3lstm_1/while/MatMul_5/Enter*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
П
lstm_1/while/MatMul_5/EnterEnterlstm_1/strided_slice_5*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context
z
lstm_1/while/add_3Addlstm_1/while/BiasAdd_1lstm_1/while/MatMul_5*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/mul_1/xConst^lstm_1/while/Identity*
_output_shapes
: *
valueB
 *ЭЬL>*
dtype0
u
lstm_1/while/mul_1Mullstm_1/while/mul_1/xlstm_1/while/add_3*'
_output_shapes
:џџџџџџџџџ*
T0
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
:џџџџџџџџџ*
T0
q
lstm_1/while/Const_2Const^lstm_1/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
q
lstm_1/while/Const_3Const^lstm_1/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$lstm_1/while/clip_by_value_1/MinimumMinimumlstm_1/while/add_4lstm_1/while/Const_3*'
_output_shapes
:џџџџџџџџџ*
T0

lstm_1/while/clip_by_value_1Maximum$lstm_1/while/clip_by_value_1/Minimumlstm_1/while/Const_2*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/while/mul_2Mullstm_1/while/clip_by_value_1lstm_1/while/Identity_4*
T0*'
_output_shapes
:џџџџџџџџџ
­
lstm_1/while/MatMul_6MatMullstm_1/while/Identity_3lstm_1/while/MatMul_6/Enter*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
П
lstm_1/while/MatMul_6/EnterEnterlstm_1/strided_slice_6*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context*
T0*
is_constant(
z
lstm_1/while/add_5Addlstm_1/while/BiasAdd_2lstm_1/while/MatMul_6*
T0*'
_output_shapes
:џџџџџџџџџ
_
lstm_1/while/TanhTanhlstm_1/while/add_5*'
_output_shapes
:џџџџџџџџџ*
T0
z
lstm_1/while/mul_3Mullstm_1/while/clip_by_valuelstm_1/while/Tanh*
T0*'
_output_shapes
:џџџџџџџџџ
s
lstm_1/while/add_6Addlstm_1/while/mul_2lstm_1/while/mul_3*'
_output_shapes
:џџџџџџџџџ*
T0
­
lstm_1/while/MatMul_7MatMullstm_1/while/Identity_3lstm_1/while/MatMul_7/Enter*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
П
lstm_1/while/MatMul_7/EnterEnterlstm_1/strided_slice_7*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:**

frame_namelstm_1/while/while_context
z
lstm_1/while/add_7Addlstm_1/while/BiasAdd_3lstm_1/while/MatMul_7*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/mul_4/xConst^lstm_1/while/Identity*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
u
lstm_1/while/mul_4Mullstm_1/while/mul_4/xlstm_1/while/add_7*
T0*'
_output_shapes
:џџџџџџџџџ
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
:џџџџџџџџџ*
T0
q
lstm_1/while/Const_4Const^lstm_1/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *    
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
:џџџџџџџџџ

lstm_1/while/clip_by_value_2Maximum$lstm_1/while/clip_by_value_2/Minimumlstm_1/while/Const_4*'
_output_shapes
:џџџџџџџџџ*
T0
a
lstm_1/while/Tanh_1Tanhlstm_1/while/add_6*
T0*'
_output_shapes
:џџџџџџџџџ
~
lstm_1/while/mul_5Mullstm_1/while/clip_by_value_2lstm_1/while/Tanh_1*
T0*'
_output_shapes
:џџџџџџџџџ

0lstm_1/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV36lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/Enterlstm_1/while/Identity_1lstm_1/while/mul_5lstm_1/while/Identity_2*
_output_shapes
: *
T0*%
_class
loc:@lstm_1/while/mul_5
љ
6lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterlstm_1/TensorArray*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations *
is_constant(
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
lstm_1/while/NextIterationNextIterationlstm_1/while/add*
_output_shapes
: *
T0
b
lstm_1/while/NextIteration_1NextIterationlstm_1/while/add_9*
_output_shapes
: *
T0

lstm_1/while/NextIteration_2NextIteration0lstm_1/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
s
lstm_1/while/NextIteration_3NextIterationlstm_1/while/mul_5*
T0*'
_output_shapes
:џџџџџџџџџ
s
lstm_1/while/NextIteration_4NextIterationlstm_1/while/add_6*
T0*'
_output_shapes
:џџџџџџџџџ
O
lstm_1/while/ExitExitlstm_1/while/Switch*
_output_shapes
: *
T0
S
lstm_1/while/Exit_1Exitlstm_1/while/Switch_1*
_output_shapes
: *
T0
S
lstm_1/while/Exit_2Exitlstm_1/while/Switch_2*
_output_shapes
: *
T0
d
lstm_1/while/Exit_3Exitlstm_1/while/Switch_3*
T0*'
_output_shapes
:џџџџџџџџџ
d
lstm_1/while/Exit_4Exitlstm_1/while/Switch_4*'
_output_shapes
:џџџџџџџџџ*
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
#lstm_1/TensorArrayStack/range/deltaConst*
dtype0*
_output_shapes
: *%
_class
loc:@lstm_1/TensorArray*
value	B :
ѓ
lstm_1/TensorArrayStack/rangeRange#lstm_1/TensorArrayStack/range/start)lstm_1/TensorArrayStack/TensorArraySizeV3#lstm_1/TensorArrayStack/range/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0*%
_class
loc:@lstm_1/TensorArray

+lstm_1/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3lstm_1/TensorArraylstm_1/TensorArrayStack/rangelstm_1/while/Exit_2*%
_class
loc:@lstm_1/TensorArray*
dtype0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*$
element_shape:џџџџџџџџџ
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
:џџџџџџџџџ
l
lstm_1/transpose_1/permConst*
_output_shapes
:*!
valueB"          *
dtype0
Б
lstm_1/transpose_1	Transpose+lstm_1/TensorArrayStack/TensorArrayGatherV3lstm_1/transpose_1/perm*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
Tperm0*
T0
a
flatten_1/ShapeShapelstm_1/transpose_1*
T0*
out_type0*
_output_shapes
:
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
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes
:
Y
flatten_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
~
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
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

flatten_1/ReshapeReshapelstm_1/transpose_1flatten_1/stack*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
m
dense_1/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
_output_shapes
: *
valueB
 *:ЭН*
dtype0
_
dense_1/random_uniform/maxConst*
valueB
 *:Э=*
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
*
seed2ЎТц
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


dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0* 
_output_shapes
:


dense_1/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
	container *
shape:
*
shared_name 
О
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*!
_class
loc:@dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
}
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel* 
_output_shapes
:

\
dense_1/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
z
dense_1/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Њ
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
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
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ*
T0
X
dense_1/ReluReludense_1/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
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
dtype0
*
_output_shapes
: *
shape: 
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
dropout_1/cond/switch_fIdentitydropout_1/cond/Switch*
_output_shapes
: *
T0

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

dropout_1/cond/mulMuldropout_1/cond/mul/Switch:1dropout_1/cond/mul/y*
T0*(
_output_shapes
:џџџџџџџџџ
Б
dropout_1/cond/mul/SwitchSwitchdense_1/Reludropout_1/cond/pred_id*
T0*
_class
loc:@dense_1/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ

 dropout_1/cond/dropout/keep_probConst^dropout_1/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
n
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
T0*
out_type0*
_output_shapes
:

)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
_output_shapes
: *
valueB
 *    *
dtype0

)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
С
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
dtype0*(
_output_shapes
:џџџџџџџџџ*
seed2ќЫњ*
seedБџх)*
T0
Ї
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
У
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:џџџџџџџџџ
Е
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_1/cond/dropout/addAdd dropout_1/cond/dropout/keep_prob%dropout_1/cond/dropout/random_uniform*(
_output_shapes
:џџџџџџџџџ*
T0
t
dropout_1/cond/dropout/FloorFloordropout_1/cond/dropout/add*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_1/cond/dropout/divRealDivdropout_1/cond/mul dropout_1/cond/dropout/keep_prob*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/divdropout_1/cond/dropout/Floor*(
_output_shapes
:џџџџџџџџџ*
T0
Џ
dropout_1/cond/Switch_1Switchdense_1/Reludropout_1/cond/pred_id*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*
T0*
_class
loc:@dense_1/Relu

dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*
T0*
N**
_output_shapes
:џџџџџџџџџ: 
m
dense_2/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *qФО*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
valueB
 *qФ>*
dtype0*
_output_shapes
: 
Њ
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
T0*
dtype0* 
_output_shapes
:
*
seed2кзШ*
seedБџх)
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
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min* 
_output_shapes
:
*
T0

dense_2/kernel
VariableV2* 
_output_shapes
:
*
	container *
shape:
*
shared_name *
dtype0
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
dense_2/kernel/readIdentitydense_2/kernel* 
_output_shapes
:
*
T0*!
_class
loc:@dense_2/kernel
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
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes	
:

dense_2/MatMulMatMuldropout_1/cond/Mergedense_2/kernel/read*(
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
dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes
: : *
T0

]
dropout_2/cond/switch_tIdentitydropout_2/cond/Switch:1*
_output_shapes
: *
T0

[
dropout_2/cond/switch_fIdentitydropout_2/cond/Switch*
T0
*
_output_shapes
: 
c
dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
: *
T0

s
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?

dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*
T0*(
_output_shapes
:џџџџџџџџџ
Б
dropout_2/cond/mul/SwitchSwitchdense_2/Reludropout_2/cond/pred_id*
T0*
_class
loc:@dense_2/Relu*<
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
)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
_output_shapes
: *
valueB
 *    *
dtype0

)dropout_2/cond/dropout/random_uniform/maxConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
С
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
seedБџх)*
T0*
dtype0*(
_output_shapes
:џџџџџџџџџ*
seed2ъУЋ
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
%dropout_2/cond/dropout/random_uniformAdd)dropout_2/cond/dropout/random_uniform/mul)dropout_2/cond/dropout/random_uniform/min*(
_output_shapes
:џџџџџџџџџ*
T0

dropout_2/cond/dropout/addAdd dropout_2/cond/dropout/keep_prob%dropout_2/cond/dropout/random_uniform*(
_output_shapes
:џџџџџџџџџ*
T0
t
dropout_2/cond/dropout/FloorFloordropout_2/cond/dropout/add*
T0*(
_output_shapes
:џџџџџџџџџ
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
dropout_2/cond/Switch_1Switchdense_2/Reludropout_2/cond/pred_id*
T0*
_class
loc:@dense_2/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ

dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul*
T0*
N**
_output_shapes
:џџџџџџџџџ: 
m
dense_3/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
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
Љ
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
seedБџх)*
T0*
dtype0*
_output_shapes
:	*
seed2Д
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
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
T0*
_output_shapes
:	

dense_3/kernel
VariableV2*
shape:	*
shared_name *
dtype0*
_output_shapes
:	*
	container 
Н
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes
:	
|
dense_3/kernel/readIdentitydense_3/kernel*
_output_shapes
:	*
T0*!
_class
loc:@dense_3/kernel
Z
dense_3/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
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
dense_3/MatMulMatMuldropout_2/cond/Mergedense_3/kernel/read*'
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
Adam/iterations/initial_valueConst*
_output_shapes
: *
value	B	 R *
dtype0	
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
Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/lr
^
Adam/lr/readIdentityAdam/lr*
T0*
_class
loc:@Adam/lr*
_output_shapes
: 
^
Adam/beta_1/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?
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
Adam/beta_1/readIdentityAdam/beta_1*
_class
loc:@Adam/beta_1*
_output_shapes
: *
T0
^
Adam/beta_2/initial_valueConst*
valueB
 *wО?*
dtype0*
_output_shapes
: 
o
Adam/beta_2
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
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
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
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
Adam/decay*
T0*
_class
loc:@Adam/decay*
_output_shapes
: 

dense_3_targetPlaceholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
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
loss/dense_3_loss/SumSumdense_3/Softmax'loss/dense_3_loss/Sum/reduction_indices*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0*
T0
~
loss/dense_3_loss/truedivRealDivdense_3/Softmaxloss/dense_3_loss/Sum*'
_output_shapes
:џџџџџџџџџ*
T0
\
loss/dense_3_loss/ConstConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
\
loss/dense_3_loss/sub/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
o
loss/dense_3_loss/subSubloss/dense_3_loss/sub/xloss/dense_3_loss/Const*
_output_shapes
: *
T0

'loss/dense_3_loss/clip_by_value/MinimumMinimumloss/dense_3_loss/truedivloss/dense_3_loss/sub*'
_output_shapes
:џџџџџџџџџ*
T0
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
loss/dense_3_loss/Sum_1Sumloss/dense_3_loss/mul)loss/dense_3_loss/Sum_1/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( *

Tidx0
c
loss/dense_3_loss/NegNegloss/dense_3_loss/Sum_1*#
_output_shapes
:џџџџџџџџџ*
T0
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
loss/dense_3_loss/mul_1Mulloss/dense_3_loss/Meandense_3_sample_weights*#
_output_shapes
:џџџџџџџџџ*
T0
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
loss/dense_3_loss/CastCastloss/dense_3_loss/NotEqual*

SrcT0
*
Truncate( *#
_output_shapes
:џџџџџџџџџ*

DstT0
c
loss/dense_3_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss/dense_3_loss/Mean_1Meanloss/dense_3_loss/Castloss/dense_3_loss/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
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
loss/dense_3_loss/Mean_2Meanloss/dense_3_loss/truediv_1loss/dense_3_loss/Const_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
O

loss/mul/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
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
metrics/acc/ArgMax_1ArgMaxdense_3/Softmaxmetrics/acc/ArgMax_1/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:џџџџџџџџџ
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
metrics/acc/MeanMeanmetrics/acc/Castmetrics/acc/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
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
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
_output_shapes
: *
T0*
_class
loc:@loss/mul*

index_type0

training/Adam/gradients/f_countConst*
_output_shapes
: *&
_class
loc:@lstm_1/while/Exit_2*
value	B : *
dtype0
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
training/Adam/gradients/SwitchSwitchtraining/Adam/gradients/Mergelstm_1/while/LoopCond*&
_class
loc:@lstm_1/while/Exit_2*
_output_shapes
: : *
T0
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
%training/Adam/gradients/NextIterationNextIterationtraining/Adam/gradients/AddH^training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPushV2H^training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPushV2l^training/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2R^training/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPushV2_1R^training/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPushV2R^training/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPushV2_1R^training/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPushV2R^training/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPushV2_1R^training/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPushV2_1R^training/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPushV2_1R^training/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPushV2b^training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPushV2V^training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPushV2d^training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPushV2X^training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPushV2\^training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPushV2S^training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPushV2d^training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPushV2X^training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPushV2\^training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPushV2S^training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPushV2Z^training/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPushV2Q^training/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPushV2R^training/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPushV2@^training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPushV2R^training/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPushV2_1@^training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPushV2B^training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPushV2R^training/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPushV2_1@^training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPushV2B^training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPushV2R^training/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPushV2@^training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPushV2R^training/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPushV2T^training/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPushV2_1@^training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPushV2B^training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPushV2P^training/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPushV2>^training/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPushV2*
T0*&
_class
loc:@lstm_1/while/Exit_2*
_output_shapes
: 
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
!training/Adam/gradients/b_count_1Enter!training/Adam/gradients/f_count_2*
T0*&
_class
loc:@lstm_1/while/Exit_2*
parallel_iterations *
is_constant( *
_output_shapes
: *B

frame_name42training/Adam/gradients/lstm_1/while/while_context
Ш
training/Adam/gradients/Merge_1Merge!training/Adam/gradients/b_count_1'training/Adam/gradients/NextIteration_1*
N*
_output_shapes
: : *
T0*&
_class
loc:@lstm_1/while/Exit_2
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
!training/Adam/gradients/b_count_2LoopCond$training/Adam/gradients/GreaterEqual*&
_class
loc:@lstm_1/while/Exit_2*
_output_shapes
: 
Й
 training/Adam/gradients/Switch_1Switchtraining/Adam/gradients/Merge_1!training/Adam/gradients/b_count_2*
T0*&
_class
loc:@lstm_1/while/Exit_2*
_output_shapes
: : 
Л
training/Adam/gradients/SubSub"training/Adam/gradients/Switch_1:1*training/Adam/gradients/GreaterEqual/Enter*
_output_shapes
: *
T0*&
_class
loc:@lstm_1/while/Exit_2
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
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/dense_3_loss/Mean_2*
_class
loc:@loss/mul*
_output_shapes
: *
T0
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
=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Ctraining/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Reshape/shape*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*
Tshape0*
_output_shapes
:
У
;training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/ShapeShapeloss/dense_3_loss/truediv_1*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*
out_type0
Ћ
:training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/TileTile=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Reshape;training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Shape*

Tmultiples0*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*#
_output_shapes
:џџџџџџџџџ
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
;training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/ConstConst*+
_class!
loc:@loss/dense_3_loss/Mean_2*
valueB: *
dtype0*
_output_shapes
:
Љ
:training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/ProdProd=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Shape_1;training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Const*+
_class!
loc:@loss/dense_3_loss/Mean_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
Д
=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Const_1Const*+
_class!
loc:@loss/dense_3_loss/Mean_2*
valueB: *
dtype0*
_output_shapes
:
­
<training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Prod_1Prod=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Shape_2=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Const_1*
	keep_dims( *

Tidx0*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*
_output_shapes
: 
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
>training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/floordivFloorDiv:training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Prod=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Maximum*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_2*
_output_shapes
: 
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
>training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/ShapeShapeloss/dense_3_loss/mul_1*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
out_type0*
_output_shapes
:
Г
@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Shape_1Const*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
valueB *
dtype0*
_output_shapes
: 
ж
Ntraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgs>training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Shape@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Shape_1*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDivRealDiv=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/truedivloss/dense_3_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1
Х
<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/SumSum@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDivNtraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
_output_shapes
:
Е
@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/ReshapeReshape<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Sum>training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Shape*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
Tshape0*#
_output_shapes
:џџџџџџџџџ
К
<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/NegNegloss/dense_3_loss/mul_1*.
_class$
" loc:@loss/dense_3_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ*
T0

Btraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDiv_1RealDiv<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Negloss/dense_3_loss/Mean_1*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ
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
>training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Sum_1Sum<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/mulPtraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/BroadcastGradientArgs:1*
T0*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
_output_shapes
:*
	keep_dims( *

Tidx0
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
Jtraining/Adam/gradients/loss/dense_3_loss/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Shape<training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Shape_1*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ѓ
8training/Adam/gradients/loss/dense_3_loss/mul_1_grad/MulMul@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Reshapedense_3_sample_weights*#
_output_shapes
:џџџџџџџџџ*
T0**
_class 
loc:@loss/dense_3_loss/mul_1
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
:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Sum_1Sum:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Mul_1Ltraining/Adam/gradients/loss/dense_3_loss/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*
_output_shapes
:
Ћ
>training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Reshape_1Reshape:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Sum_1<training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Shape_1*#
_output_shapes
:џџџџџџџџџ*
T0**
_class 
loc:@loss/dense_3_loss/mul_1*
Tshape0
Й
9training/Adam/gradients/loss/dense_3_loss/Mean_grad/ShapeShapeloss/dense_3_loss/Neg*
_output_shapes
:*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
out_type0
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
?training/Adam/gradients/loss/dense_3_loss/Mean_grad/range/deltaConst*
_output_shapes
: *)
_class
loc:@loss/dense_3_loss/Mean*
value	B :*
dtype0
б
9training/Adam/gradients/loss/dense_3_loss/Mean_grad/rangeRange?training/Adam/gradients/loss/dense_3_loss/Mean_grad/range/start8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Size?training/Adam/gradients/loss/dense_3_loss/Mean_grad/range/delta*

Tidx0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
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
N*
_output_shapes
:*
T0*)
_class
loc:@loss/dense_3_loss/Mean
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
;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_2Shapeloss/dense_3_loss/Neg*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
out_type0*
_output_shapes
:
М
;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_3Shapeloss/dense_3_loss/Mean*)
_class
loc:@loss/dense_3_loss/Mean*
out_type0*
_output_shapes
:*
T0
Ў
9training/Adam/gradients/loss/dense_3_loss/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*)
_class
loc:@loss/dense_3_loss/Mean*
valueB: 
Ё
8training/Adam/gradients/loss/dense_3_loss/Mean_grad/ProdProd;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_29training/Adam/gradients/loss/dense_3_loss/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*)
_class
loc:@loss/dense_3_loss/Mean
А
;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Const_1Const*)
_class
loc:@loss/dense_3_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
Ѕ
:training/Adam/gradients/loss/dense_3_loss/Mean_grad/Prod_1Prod;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_3;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Const_1*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: *
	keep_dims( *

Tidx0
Ќ
?training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum_1/yConst*
_output_shapes
: *)
_class
loc:@loss/dense_3_loss/Mean*
value	B :*
dtype0

=training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum_1Maximum:training/Adam/gradients/loss/dense_3_loss/Mean_grad/Prod_1?training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum_1/y*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
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
;training/Adam/gradients/loss/dense_3_loss/Mean_grad/truedivRealDiv8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Tile8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Cast*#
_output_shapes
:џџџџџџџџџ*
T0*)
_class
loc:@loss/dense_3_loss/Mean
в
6training/Adam/gradients/loss/dense_3_loss/Neg_grad/NegNeg;training/Adam/gradients/loss/dense_3_loss/Mean_grad/truediv*
T0*(
_class
loc:@loss/dense_3_loss/Neg*#
_output_shapes
:џџџџџџџџџ
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
8training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/addAdd)loss/dense_3_loss/Sum_1/reduction_indices9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Size*
T0**
_class 
loc:@loss/dense_3_loss/Sum_1*
_output_shapes
: 

8training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/modFloorMod8training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/add9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Size**
_class 
loc:@loss/dense_3_loss/Sum_1*
_output_shapes
: *
T0
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
@training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/range/deltaConst*
dtype0*
_output_shapes
: **
_class 
loc:@loss/dense_3_loss/Sum_1*
value	B :
ж
:training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/rangeRange@training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/range/start9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Size@training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/range/delta*

Tidx0**
_class 
loc:@loss/dense_3_loss/Sum_1*
_output_shapes
:
­
?training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Fill/valueConst*
_output_shapes
: **
_class 
loc:@loss/dense_3_loss/Sum_1*
value	B :*
dtype0

9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/FillFill<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Shape_1?training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Fill/value**
_class 
loc:@loss/dense_3_loss/Sum_1*

index_type0*
_output_shapes
: *
T0

Btraining/Adam/gradients/loss/dense_3_loss/Sum_1_grad/DynamicStitchDynamicStitch:training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/range8training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/mod:training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Shape9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Fill**
_class 
loc:@loss/dense_3_loss/Sum_1*
N*
_output_shapes
:*
T0
Ќ
>training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Maximum/yConst*
_output_shapes
: **
_class 
loc:@loss/dense_3_loss/Sum_1*
value	B :*
dtype0

<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/MaximumMaximumBtraining/Adam/gradients/loss/dense_3_loss/Sum_1_grad/DynamicStitch>training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Maximum/y*
_output_shapes
:*
T0**
_class 
loc:@loss/dense_3_loss/Sum_1

=training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/floordivFloorDiv:training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Shape<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Maximum*
T0**
_class 
loc:@loss/dense_3_loss/Sum_1*
_output_shapes
:
И
<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/ReshapeReshape6training/Adam/gradients/loss/dense_3_loss/Neg_grad/NegBtraining/Adam/gradients/loss/dense_3_loss/Sum_1_grad/DynamicStitch*
T0**
_class 
loc:@loss/dense_3_loss/Sum_1*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ў
9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/TileTile<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Reshape=training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/floordiv**
_class 
loc:@loss/dense_3_loss/Sum_1*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
А
8training/Adam/gradients/loss/dense_3_loss/mul_grad/ShapeShapedense_3_target*
T0*(
_class
loc:@loss/dense_3_loss/mul*
out_type0*
_output_shapes
:
Й
:training/Adam/gradients/loss/dense_3_loss/mul_grad/Shape_1Shapeloss/dense_3_loss/Log*(
_class
loc:@loss/dense_3_loss/mul*
out_type0*
_output_shapes
:*
T0
О
Htraining/Adam/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8training/Adam/gradients/loss/dense_3_loss/mul_grad/Shape:training/Adam/gradients/loss/dense_3_loss/mul_grad/Shape_1*
T0*(
_class
loc:@loss/dense_3_loss/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ы
6training/Adam/gradients/loss/dense_3_loss/mul_grad/MulMul9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Tileloss/dense_3_loss/Log*(
_class
loc:@loss/dense_3_loss/mul*'
_output_shapes
:џџџџџџџџџ*
T0
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
8training/Adam/gradients/loss/dense_3_loss/mul_grad/Mul_1Muldense_3_target9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Tile*(
_class
loc:@loss/dense_3_loss/mul*'
_output_shapes
:џџџџџџџџџ*
T0
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
Reciprocalloss/dense_3_loss/clip_by_value=^training/Adam/gradients/loss/dense_3_loss/mul_grad/Reshape_1*(
_class
loc:@loss/dense_3_loss/Log*'
_output_shapes
:џџџџџџџџџ*
T0
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
Ctraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/SelectSelectItraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/GreaterEqual6training/Adam/gradients/loss/dense_3_loss/Log_grad/mulBtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/zeros*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*'
_output_shapes
:џџџџџџџџџ
ќ
Etraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Select_1SelectItraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/GreaterEqualBtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/zeros6training/Adam/gradients/loss/dense_3_loss/Log_grad/mul*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*'
_output_shapes
:џџџџџџџџџ
д
@training/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/SumSumCtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/SelectRtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
_output_shapes
:
Щ
Dtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/ReshapeReshape@training/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/SumBtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape*
T0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
Tshape0*'
_output_shapes
:џџџџџџџџџ
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
Ftraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Reshape_1ReshapeBtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Sum_1Dtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape_1*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
Tshape0*
_output_shapes
: *
T0
п
Jtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/ShapeShapeloss/dense_3_loss/truediv*
_output_shapes
:*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
out_type0
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
Jtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/zerosFillLtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape_2Ptraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/zeros/Const*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*

index_type0
ћ
Ntraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/LessEqual	LessEqualloss/dense_3_loss/truedivloss/dense_3_loss/sub*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ*
T0

Ztraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/ShapeLtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape_1*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ѕ
Ktraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/SelectSelectNtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/LessEqualDtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/ReshapeJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/zeros*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ*
T0
Ї
Mtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Select_1SelectNtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/LessEqualJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/zerosDtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum
є
Htraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/SumSumKtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/SelectZtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
_output_shapes
:*
	keep_dims( *

Tidx0
щ
Ltraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/ReshapeReshapeHtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/SumJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
Tshape0*'
_output_shapes
:џџџџџџџџџ
њ
Jtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Sum_1SumMtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Select_1\training/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
_output_shapes
:*
	keep_dims( *

Tidx0
о
Ntraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Reshape_1ReshapeJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Sum_1Ltraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape_1*
T0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
Tshape0*
_output_shapes
: 
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
Ltraining/Adam/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs<training/Adam/gradients/loss/dense_3_loss/truediv_grad/Shape>training/Adam/gradients/loss/dense_3_loss/truediv_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv

>training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDivRealDivLtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Reshapeloss/dense_3_loss/Sum*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*'
_output_shapes
:џџџџџџџџџ
Н
:training/Adam/gradients/loss/dense_3_loss/truediv_grad/SumSum>training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDivLtraining/Adam/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgs*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
:*
	keep_dims( *

Tidx0
Б
>training/Adam/gradients/loss/dense_3_loss/truediv_grad/ReshapeReshape:training/Adam/gradients/loss/dense_3_loss/truediv_grad/Sum<training/Adam/gradients/loss/dense_3_loss/truediv_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*
Tshape0
В
:training/Adam/gradients/loss/dense_3_loss/truediv_grad/NegNegdense_3/Softmax*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*'
_output_shapes
:џџџџџџџџџ
ў
@training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDiv_1RealDiv:training/Adam/gradients/loss/dense_3_loss/truediv_grad/Negloss/dense_3_loss/Sum*,
_class"
 loc:@loss/dense_3_loss/truediv*'
_output_shapes
:џџџџџџџџџ*
T0

@training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDiv_2RealDiv@training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDiv_1loss/dense_3_loss/Sum*,
_class"
 loc:@loss/dense_3_loss/truediv*'
_output_shapes
:џџџџџџџџџ*
T0
Б
:training/Adam/gradients/loss/dense_3_loss/truediv_grad/mulMulLtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Reshape@training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDiv_2*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*'
_output_shapes
:џџџџџџџџџ
Н
<training/Adam/gradients/loss/dense_3_loss/truediv_grad/Sum_1Sum:training/Adam/gradients/loss/dense_3_loss/truediv_grad/mulNtraining/Adam/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv
З
@training/Adam/gradients/loss/dense_3_loss/truediv_grad/Reshape_1Reshape<training/Adam/gradients/loss/dense_3_loss/truediv_grad/Sum_1>training/Adam/gradients/loss/dense_3_loss/truediv_grad/Shape_1*,
_class"
 loc:@loss/dense_3_loss/truediv*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
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
:training/Adam/gradients/loss/dense_3_loss/Sum_grad/Shape_1Const*(
_class
loc:@loss/dense_3_loss/Sum*
valueB *
dtype0*
_output_shapes
: 
Њ
>training/Adam/gradients/loss/dense_3_loss/Sum_grad/range/startConst*(
_class
loc:@loss/dense_3_loss/Sum*
value	B : *
dtype0*
_output_shapes
: 
Њ
>training/Adam/gradients/loss/dense_3_loss/Sum_grad/range/deltaConst*(
_class
loc:@loss/dense_3_loss/Sum*
value	B :*
dtype0*
_output_shapes
: 
Ь
8training/Adam/gradients/loss/dense_3_loss/Sum_grad/rangeRange>training/Adam/gradients/loss/dense_3_loss/Sum_grad/range/start7training/Adam/gradients/loss/dense_3_loss/Sum_grad/Size>training/Adam/gradients/loss/dense_3_loss/Sum_grad/range/delta*
_output_shapes
:*

Tidx0*(
_class
loc:@loss/dense_3_loss/Sum
Љ
=training/Adam/gradients/loss/dense_3_loss/Sum_grad/Fill/valueConst*(
_class
loc:@loss/dense_3_loss/Sum*
value	B :*
dtype0*
_output_shapes
: 

7training/Adam/gradients/loss/dense_3_loss/Sum_grad/FillFill:training/Adam/gradients/loss/dense_3_loss/Sum_grad/Shape_1=training/Adam/gradients/loss/dense_3_loss/Sum_grad/Fill/value*
_output_shapes
: *
T0*(
_class
loc:@loss/dense_3_loss/Sum*

index_type0
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
;training/Adam/gradients/loss/dense_3_loss/Sum_grad/floordivFloorDiv8training/Adam/gradients/loss/dense_3_loss/Sum_grad/Shape:training/Adam/gradients/loss/dense_3_loss/Sum_grad/Maximum*
_output_shapes
:*
T0*(
_class
loc:@loss/dense_3_loss/Sum
М
:training/Adam/gradients/loss/dense_3_loss/Sum_grad/ReshapeReshape@training/Adam/gradients/loss/dense_3_loss/truediv_grad/Reshape_1@training/Adam/gradients/loss/dense_3_loss/Sum_grad/DynamicStitch*(
_class
loc:@loss/dense_3_loss/Sum*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
І
7training/Adam/gradients/loss/dense_3_loss/Sum_grad/TileTile:training/Adam/gradients/loss/dense_3_loss/Sum_grad/Reshape;training/Adam/gradients/loss/dense_3_loss/Sum_grad/floordiv*(
_class
loc:@loss/dense_3_loss/Sum*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0

training/Adam/gradients/AddNAddN>training/Adam/gradients/loss/dense_3_loss/truediv_grad/Reshape7training/Adam/gradients/loss/dense_3_loss/Sum_grad/Tile*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*
N*'
_output_shapes
:џџџџџџџџџ
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
2training/Adam/gradients/dense_3/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_3/Softmax_grad/mul_1dense_3/kernel/read*
T0*!
_class
loc:@dense_3/MatMul*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
ћ
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldropout_2/cond/Merge2training/Adam/gradients/dense_3/Softmax_grad/mul_1*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_3/MatMul
ћ
;training/Adam/gradients/dropout_2/cond/Merge_grad/cond_gradSwitch2training/Adam/gradients/dense_3/MatMul_grad/MatMuldropout_2/cond/pred_id*!
_class
loc:@dense_3/MatMul*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*
T0
И
 training/Adam/gradients/Switch_2Switchdense_2/Reludropout_2/cond/pred_id*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*
T0*
_class
loc:@dense_2/Relu
Є
 training/Adam/gradients/IdentityIdentity"training/Adam/gradients/Switch_2:1*(
_output_shapes
:џџџџџџџџџ*
T0*
_class
loc:@dense_2/Relu
Ђ
training/Adam/gradients/Shape_1Shape"training/Adam/gradients/Switch_2:1*
_class
loc:@dense_2/Relu*
out_type0*
_output_shapes
:*
T0
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
training/Adam/gradients/zerosFilltraining/Adam/gradients/Shape_1#training/Adam/gradients/zeros/Const*(
_output_shapes
:џџџџџџџџџ*
T0*
_class
loc:@dense_2/Relu*

index_type0

>training/Adam/gradients/dropout_2/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_2/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros*
T0*
_class
loc:@dense_2/Relu*
N**
_output_shapes
:џџџџџџџџџ: 
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
Mtraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/MulMul=training/Adam/gradients/dropout_2/cond/Merge_grad/cond_grad:1dropout_2/cond/dropout/Floor*-
_class#
!loc:@dropout_2/cond/dropout/mul*(
_output_shapes
:џџџџџџџџџ*
T0
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
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Muldropout_2/cond/dropout/div=training/Adam/gradients/dropout_2/cond/Merge_grad/cond_grad:1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*(
_output_shapes
:џџџџџџџџџ
У
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*-
_class#
!loc:@dropout_2/cond/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
М
Atraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
Tshape0*(
_output_shapes
:џџџџџџџџџ
О
=training/Adam/gradients/dropout_2/cond/dropout/div_grad/ShapeShapedropout_2/cond/mul*
T0*-
_class#
!loc:@dropout_2/cond/dropout/div*
out_type0*
_output_shapes
:
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
?training/Adam/gradients/dropout_2/cond/dropout/div_grad/RealDivRealDiv?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Reshape dropout_2/cond/dropout/keep_prob*(
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_2/cond/dropout/div
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
7training/Adam/gradients/dropout_2/cond/mul_grad/Shape_1Const*%
_class
loc:@dropout_2/cond/mul*
valueB *
dtype0*
_output_shapes
: 
В
Etraining/Adam/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/dropout_2/cond/mul_grad/Shape7training/Adam/gradients/dropout_2/cond/mul_grad/Shape_1*%
_class
loc:@dropout_2/cond/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
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
5training/Adam/gradients/dropout_2/cond/mul_grad/Mul_1Muldropout_2/cond/mul/Switch:1?training/Adam/gradients/dropout_2/cond/dropout/div_grad/Reshape*%
_class
loc:@dropout_2/cond/mul*(
_output_shapes
:џџџџџџџџџ*
T0
Ѓ
5training/Adam/gradients/dropout_2/cond/mul_grad/Sum_1Sum5training/Adam/gradients/dropout_2/cond/mul_grad/Mul_1Gtraining/Adam/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@dropout_2/cond/mul

9training/Adam/gradients/dropout_2/cond/mul_grad/Reshape_1Reshape5training/Adam/gradients/dropout_2/cond/mul_grad/Sum_17training/Adam/gradients/dropout_2/cond/mul_grad/Shape_1*
_output_shapes
: *
T0*%
_class
loc:@dropout_2/cond/mul*
Tshape0
И
 training/Adam/gradients/Switch_3Switchdense_2/Reludropout_2/cond/pred_id*
_class
loc:@dense_2/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*
T0
Є
"training/Adam/gradients/Identity_1Identity training/Adam/gradients/Switch_3*
T0*
_class
loc:@dense_2/Relu*(
_output_shapes
:џџџџџџџџџ
 
training/Adam/gradients/Shape_2Shape training/Adam/gradients/Switch_3*
_output_shapes
:*
T0*
_class
loc:@dense_2/Relu*
out_type0
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
training/Adam/gradients/zeros_1Filltraining/Adam/gradients/Shape_2%training/Adam/gradients/zeros_1/Const*
_class
loc:@dense_2/Relu*

index_type0*(
_output_shapes
:џџџџџџџџџ*
T0

@training/Adam/gradients/dropout_2/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_17training/Adam/gradients/dropout_2/cond/mul_grad/Reshape*
_class
loc:@dense_2/Relu*
N**
_output_shapes
:џџџџџџџџџ: *
T0

training/Adam/gradients/AddN_1AddN>training/Adam/gradients/dropout_2/cond/Switch_1_grad/cond_grad@training/Adam/gradients/dropout_2/cond/mul/Switch_grad/cond_grad*
N*(
_output_shapes
:џџџџџџџџџ*
T0*
_class
loc:@dense_2/Relu
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
4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1MatMuldropout_1/cond/Merge2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
transpose_b( *
T0*!
_class
loc:@dense_2/MatMul* 
_output_shapes
:
*
transpose_a(
ћ
;training/Adam/gradients/dropout_1/cond/Merge_grad/cond_gradSwitch2training/Adam/gradients/dense_2/MatMul_grad/MatMuldropout_1/cond/pred_id*
T0*!
_class
loc:@dense_2/MatMul*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
И
 training/Adam/gradients/Switch_4Switchdense_1/Reludropout_1/cond/pred_id*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*
T0*
_class
loc:@dense_1/Relu
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
training/Adam/gradients/zeros_2Filltraining/Adam/gradients/Shape_3%training/Adam/gradients/zeros_2/Const*
_class
loc:@dense_1/Relu*

index_type0*(
_output_shapes
:џџџџџџџџџ*
T0

>training/Adam/gradients/dropout_1/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_1/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros_2*
T0*
_class
loc:@dense_1/Relu*
N**
_output_shapes
:џџџџџџџџџ: 
Ц
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ShapeShapedropout_1/cond/dropout/div*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
out_type0*
_output_shapes
:
Ъ
?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1Shapedropout_1/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
out_type0*
_output_shapes
:
в
Mtraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/MulMul=training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad:1dropout_1/cond/dropout/Floor*-
_class#
!loc:@dropout_1/cond/dropout/mul*(
_output_shapes
:џџџџџџџџџ*
T0
Н
;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:
Ж
?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
Tshape0*(
_output_shapes
:џџџџџџџџџ

=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Muldropout_1/cond/dropout/div=training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad:1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*(
_output_shapes
:џџџџџџџџџ
У
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
М
Atraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*(
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
Tshape0
О
=training/Adam/gradients/dropout_1/cond/dropout/div_grad/ShapeShapedropout_1/cond/mul*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*
out_type0*
_output_shapes
:
Б
?training/Adam/gradients/dropout_1/cond/dropout/div_grad/Shape_1Const*-
_class#
!loc:@dropout_1/cond/dropout/div*
valueB *
dtype0*
_output_shapes
: 
в
Mtraining/Adam/gradients/dropout_1/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_1/cond/dropout/div_grad/Shape?training/Adam/gradients/dropout_1/cond/dropout/div_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div

?training/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDivRealDiv?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Reshape dropout_1/cond/dropout/keep_prob*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ
С
;training/Adam/gradients/dropout_1/cond/dropout/div_grad/SumSum?training/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDivMtraining/Adam/gradients/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*
_output_shapes
:
Ж
?training/Adam/gradients/dropout_1/cond/dropout/div_grad/ReshapeReshape;training/Adam/gradients/dropout_1/cond/dropout/div_grad/Sum=training/Adam/gradients/dropout_1/cond/dropout/div_grad/Shape*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*
Tshape0*(
_output_shapes
:џџџџџџџџџ
И
;training/Adam/gradients/dropout_1/cond/dropout/div_grad/NegNegdropout_1/cond/mul*-
_class#
!loc:@dropout_1/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ*
T0

Atraining/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDiv_1RealDiv;training/Adam/gradients/dropout_1/cond/dropout/div_grad/Neg dropout_1/cond/dropout/keep_prob*-
_class#
!loc:@dropout_1/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ*
T0

Atraining/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDiv_2RealDivAtraining/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDiv_1 dropout_1/cond/dropout/keep_prob*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*(
_output_shapes
:џџџџџџџџџ
Ј
;training/Adam/gradients/dropout_1/cond/dropout/div_grad/mulMul?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ReshapeAtraining/Adam/gradients/dropout_1/cond/dropout/div_grad/RealDiv_2*(
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div
С
=training/Adam/gradients/dropout_1/cond/dropout/div_grad/Sum_1Sum;training/Adam/gradients/dropout_1/cond/dropout/div_grad/mulOtraining/Adam/gradients/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*
_output_shapes
:
Њ
Atraining/Adam/gradients/dropout_1/cond/dropout/div_grad/Reshape_1Reshape=training/Adam/gradients/dropout_1/cond/dropout/div_grad/Sum_1?training/Adam/gradients/dropout_1/cond/dropout/div_grad/Shape_1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/div*
Tshape0*
_output_shapes
: 
З
5training/Adam/gradients/dropout_1/cond/mul_grad/ShapeShapedropout_1/cond/mul/Switch:1*
_output_shapes
:*
T0*%
_class
loc:@dropout_1/cond/mul*
out_type0
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
ы
3training/Adam/gradients/dropout_1/cond/mul_grad/MulMul?training/Adam/gradients/dropout_1/cond/dropout/div_grad/Reshapedropout_1/cond/mul/y*
T0*%
_class
loc:@dropout_1/cond/mul*(
_output_shapes
:џџџџџџџџџ

3training/Adam/gradients/dropout_1/cond/mul_grad/SumSum3training/Adam/gradients/dropout_1/cond/mul_grad/MulEtraining/Adam/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgs*
T0*%
_class
loc:@dropout_1/cond/mul*
_output_shapes
:*
	keep_dims( *

Tidx0

7training/Adam/gradients/dropout_1/cond/mul_grad/ReshapeReshape3training/Adam/gradients/dropout_1/cond/mul_grad/Sum5training/Adam/gradients/dropout_1/cond/mul_grad/Shape*
T0*%
_class
loc:@dropout_1/cond/mul*
Tshape0*(
_output_shapes
:џџџџџџџџџ
є
5training/Adam/gradients/dropout_1/cond/mul_grad/Mul_1Muldropout_1/cond/mul/Switch:1?training/Adam/gradients/dropout_1/cond/dropout/div_grad/Reshape*(
_output_shapes
:џџџџџџџџџ*
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
9training/Adam/gradients/dropout_1/cond/mul_grad/Reshape_1Reshape5training/Adam/gradients/dropout_1/cond/mul_grad/Sum_17training/Adam/gradients/dropout_1/cond/mul_grad/Shape_1*
_output_shapes
: *
T0*%
_class
loc:@dropout_1/cond/mul*
Tshape0
И
 training/Adam/gradients/Switch_5Switchdense_1/Reludropout_1/cond/pred_id*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*
T0*
_class
loc:@dense_1/Relu
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
@training/Adam/gradients/dropout_1/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_37training/Adam/gradients/dropout_1/cond/mul_grad/Reshape*
T0*
_class
loc:@dense_1/Relu*
N**
_output_shapes
:џџџџџџџџџ: 

training/Adam/gradients/AddN_2AddN>training/Adam/gradients/dropout_1/cond/Switch_1_grad/cond_grad@training/Adam/gradients/dropout_1/cond/mul/Switch_grad/cond_grad*
_class
loc:@dense_1/Relu*
N*(
_output_shapes
:џџџџџџџџџ*
T0
Р
2training/Adam/gradients/dense_1/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_2dense_1/Relu*
_class
loc:@dense_1/Relu*(
_output_shapes
:џџџџџџџџџ*
T0
м
8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_1/Relu_grad/ReluGrad*
T0*"
_class
loc:@dense_1/BiasAdd*
data_formatNHWC*
_output_shapes	
:

2training/Adam/gradients/dense_1/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_1/Relu_grad/ReluGraddense_1/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_1/MatMul
љ
4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1MatMulflatten_1/Reshape2training/Adam/gradients/dense_1/Relu_grad/ReluGrad*
transpose_b( *
T0*!
_class
loc:@dense_1/MatMul* 
_output_shapes
:
*
transpose_a(
Ќ
4training/Adam/gradients/flatten_1/Reshape_grad/ShapeShapelstm_1/transpose_1*
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
 :џџџџџџџџџџџџџџџџџџ
Л
Atraining/Adam/gradients/lstm_1/transpose_1_grad/InvertPermutationInvertPermutationlstm_1/transpose_1/perm*
T0*%
_class
loc:@lstm_1/transpose_1*
_output_shapes
:
Д
9training/Adam/gradients/lstm_1/transpose_1_grad/transpose	Transpose6training/Adam/gradients/flatten_1/Reshape_grad/ReshapeAtraining/Adam/gradients/lstm_1/transpose_1_grad/InvertPermutation*
T0*%
_class
loc:@lstm_1/transpose_1*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
Tperm0

jtraining/Adam/gradients/lstm_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lstm_1/TensorArraylstm_1/while/Exit_2*%
_class
loc:@lstm_1/TensorArray*#
sourcetraining/Adam/gradients*
_output_shapes

:: 
М
ftraining/Adam/gradients/lstm_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentitylstm_1/while/Exit_2k^training/Adam/gradients/lstm_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*%
_class
loc:@lstm_1/TensorArray
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
:џџџџџџџџџ
x
$training/Adam/gradients/zeros_like_1	ZerosLikelstm_1/while/Exit_4*
T0*'
_output_shapes
:џџџџџџџџџ
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
:џџџџџџџџџ*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*&
_class
loc:@lstm_1/while/Exit_3*
parallel_iterations 
В
7training/Adam/gradients/lstm_1/while/Exit_4_grad/b_exitEnter$training/Adam/gradients/zeros_like_1*
is_constant( *'
_output_shapes
:џџџџџџџџџ*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*&
_class
loc:@lstm_1/while/Exit_4*
parallel_iterations 

;training/Adam/gradients/lstm_1/while/Switch_2_grad/b_switchMerge7training/Adam/gradients/lstm_1/while/Exit_2_grad/b_exitBtraining/Adam/gradients/lstm_1/while/Switch_2_grad_1/NextIteration*
_output_shapes
: : *
T0*'
_class
loc:@lstm_1/while/Merge_2*
N
Ї
;training/Adam/gradients/lstm_1/while/Switch_3_grad/b_switchMerge7training/Adam/gradients/lstm_1/while/Exit_3_grad/b_exitBtraining/Adam/gradients/lstm_1/while/Switch_3_grad_1/NextIteration*
T0*'
_class
loc:@lstm_1/while/Merge_3*
N*)
_output_shapes
:џџџџџџџџџ: 
Ї
;training/Adam/gradients/lstm_1/while/Switch_4_grad/b_switchMerge7training/Adam/gradients/lstm_1/while/Exit_4_grad/b_exitBtraining/Adam/gradients/lstm_1/while/Switch_4_grad_1/NextIteration*
T0*'
_class
loc:@lstm_1/while/Merge_4*
N*)
_output_shapes
:џџџџџџџџџ: 
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
&:џџџџџџџџџ:џџџџџџџџџ

8training/Adam/gradients/lstm_1/while/Merge_4_grad/SwitchSwitch;training/Adam/gradients/lstm_1/while/Switch_4_grad/b_switch!training/Adam/gradients/b_count_2*
T0*'
_class
loc:@lstm_1/while/Merge_4*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
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
:џџџџџџџџџ
г
6training/Adam/gradients/lstm_1/while/Enter_4_grad/ExitExit8training/Adam/gradients/lstm_1/while/Merge_4_grad/Switch*
T0*'
_class
loc:@lstm_1/while/Enter_4*'
_output_shapes
:џџџџџџџџџ
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
ktraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity:training/Adam/gradients/lstm_1/while/Merge_2_grad/Switch:1p^training/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*%
_class
loc:@lstm_1/while/mul_5

_training/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3otraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3jtraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2ktraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*%
_class
loc:@lstm_1/while/mul_5*
dtype0*'
_output_shapes
:џџџџџџџџџ
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
etraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2etraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*
	elem_type0*C
_class9
7loc:@lstm_1/while/Identity_1loc:@lstm_1/while/mul_5*

stack_name *
_output_shapes
:
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
ktraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2etraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterlstm_1/while/Identity_1^training/Adam/gradients/Add*
_output_shapes
: *
swap_memory(*
T0*%
_class
loc:@lstm_1/while/mul_5
и
jtraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2ptraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
: *
	elem_type0

ptraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnteretraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
р
ftraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerG^training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV2G^training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV2k^training/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Q^training/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2_1Q^training/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopV2Q^training/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2_1Q^training/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopV2Q^training/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2_1Q^training/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2_1Q^training/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2_1Q^training/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPopV2a^training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopV2U^training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopV2c^training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopV2W^training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopV2[^training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopV2R^training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopV2c^training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopV2W^training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopV2[^training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopV2R^training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopV2Y^training/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopV2P^training/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopV2Q^training/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopV2?^training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPopV2Q^training/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2_1?^training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPopV2A^training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPopV2Q^training/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2_1?^training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPopV2A^training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPopV2Q^training/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopV2?^training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPopV2Q^training/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2S^training/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2_1?^training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV2A^training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPopV2O^training/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopV2=^training/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPopV2*%
_class
loc:@lstm_1/while/mul_5
Ї
training/Adam/gradients/AddN_3AddN:training/Adam/gradients/lstm_1/while/Merge_3_grad/Switch:1_training/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
T0*'
_class
loc:@lstm_1/while/Merge_3*
N*'
_output_shapes
:џџџџџџџџџ
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
Ktraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations 
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
StackPopV2Vtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0
П
Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *%
_class
loc:@lstm_1/while/mul_5*
valueB :
џџџџџџџџџ

Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Const_1*%
_class
loc:@lstm_1/while/mul_5*

stack_name *
_output_shapes
:*
	elem_type0
Ы
Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc_1*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
ч
Straining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/mul_5_grad/Shape_1^training/Adam/gradients/Add*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
:*
swap_memory(*
T0
Ќ
Rtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
:*
	elem_type0
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
3training/Adam/gradients/lstm_1/while/mul_5_grad/MulMultraining/Adam/gradients/AddN_3>training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/mul_5
Х
9training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/ConstConst*
dtype0*
_output_shapes
: *?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_5*
valueB :
џџџџџџџџџ
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
?training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPushV2StackPushV29training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/Enterlstm_1/while/Tanh_1^training/Adam/gradients/Add*'
_output_shapes
:џџџџџџџџџ*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/mul_5

>training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV2
StackPopV2Dtraining/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_5*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
Ц
Dtraining/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV2/EnterEnter9training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_5*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context

3training/Adam/gradients/lstm_1/while/mul_5_grad/SumSum3training/Adam/gradients/lstm_1/while/mul_5_grad/MulEtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
:
А
7training/Adam/gradients/lstm_1/while/mul_5_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/mul_5_grad/SumPtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_5*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ї
5training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1Mul@training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPopV2training/Adam/gradients/AddN_3*
T0*%
_class
loc:@lstm_1/while/mul_5*'
_output_shapes
:џџџџџџџџџ
а
;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/ConstConst*
_output_shapes
: *H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_5*
valueB :
џџџџџџџџџ*
dtype0

;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/f_accStackV2;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/Const*
	elem_type0*H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_5*

stack_name *
_output_shapes
:
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
:џџџџџџџџџ*
swap_memory(

@training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPopV2
StackPopV2Ftraining/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPopV2/Enter^training/Adam/gradients/Sub*'
_output_shapes
:џџџџџџџџџ*
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
5training/Adam/gradients/lstm_1/while/mul_5_grad/Sum_1Sum5training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1Gtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/mul_5*
_output_shapes
:
Ж
9training/Adam/gradients/lstm_1/while/mul_5_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/mul_5_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopV2_1*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/mul_5*
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
Etraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/zeros/ConstConst^training/Adam/gradients/Sub*
_output_shapes
: */
_class%
#!loc:@lstm_1/while/clip_by_value_2*
valueB
 *    *
dtype0
Ц
?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/zerosFillAtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Shape_2Etraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/zeros/Const*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*

index_type0*'
_output_shapes
:џџџџџџџџџ
м
Ftraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqualGreaterEqualQtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopV2Ntraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/Const_1*'
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2
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
:џџџџџџџџџ*
swap_memory(
С
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopV2
StackPopV2Wtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopV2/Enter^training/Adam/gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
і
Wtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopV2/EnterEnterLtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0
т
Ntraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/Const_1Const^training/Adam/gradients/Sub*
dtype0*
_output_shapes
: */
_class%
#!loc:@lstm_1/while/clip_by_value_2*
valueB
 *    
ѕ
Otraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgsBroadcastGradientArgsZtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopV2Atraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Shape_1*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
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
StackPopV2`training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2
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
@training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/SelectSelectFtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual7training/Adam/gradients/lstm_1/while/mul_5_grad/Reshape?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/zeros*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*'
_output_shapes
:џџџџџџџџџ
ё
Btraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Select_1SelectFtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual?training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/zeros7training/Adam/gradients/lstm_1/while/mul_5_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
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
Atraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/ReshapeReshape=training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/SumZtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopV2*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
Tshape0*'
_output_shapes
:џџџџџџџџџ
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
9training/Adam/gradients/lstm_1/while/Tanh_1_grad/TanhGradTanhGrad>training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPopV29training/Adam/gradients/lstm_1/while/mul_5_grad/Reshape_1*&
_class
loc:@lstm_1/while/Tanh_1*'
_output_shapes
:џџџџџџџџџ*
T0
й
Btraining/Adam/gradients/lstm_1/while/Switch_2_grad_1/NextIterationNextIteration:training/Adam/gradients/lstm_1/while/Merge_2_grad/Switch:1*
T0*'
_class
loc:@lstm_1/while/Merge_2*
_output_shapes
: 
в
Gtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ShapeShapelstm_1/while/add_8*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
out_type0*
_output_shapes
:
у
Itraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_1Const^training/Adam/gradients/Sub*
_output_shapes
: *7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB *
dtype0
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
:џџџџџџџџџ
№
Ktraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual	LessEqualVtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopV2Straining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/Const_1*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*'
_output_shapes
:џџџџџџџџџ
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
Wtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPushV2StackPushV2Qtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/Enterlstm_1/while/add_8^training/Adam/gradients/Add*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*'
_output_shapes
:џџџџџџџџџ*
swap_memory(
г
Vtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopV2
StackPopV2\training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopV2/Enter^training/Adam/gradients/Sub*'
_output_shapes
:џџџџџџџџџ*
	elem_type0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum

\training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopV2/EnterEnterQtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
parallel_iterations *
is_constant(
я
Straining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/Const_1Const^training/Adam/gradients/Sub*
dtype0*
_output_shapes
: *7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB
 *  ?
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
StackPopV2htraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
_output_shapes
:*
	elem_type0
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
Htraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SelectSelectKtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqualAtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/ReshapeGtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros*'
_output_shapes
:џџџџџџџџџ*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum

Jtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Select_1SelectKtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqualGtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zerosAtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum
ш
Etraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SumSumHtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SelectWtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
_output_shapes
:*
	keep_dims( *

Tidx0
ј
Itraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ReshapeReshapeEtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Sumbtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopV2*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
Tshape0*'
_output_shapes
:џџџџџџџџџ
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
Ktraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Reshape_1ReshapeGtraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Sum_1Itraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_1*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
Tshape0*
_output_shapes
: 

training/Adam/gradients/AddN_4AddN:training/Adam/gradients/lstm_1/while/Merge_4_grad/Switch:19training/Adam/gradients/lstm_1/while/Tanh_1_grad/TanhGrad*
T0*'
_class
loc:@lstm_1/while/Merge_4*
N*'
_output_shapes
:џџџџџџџџџ
Ў
5training/Adam/gradients/lstm_1/while/add_6_grad/ShapeShapelstm_1/while/mul_2*
T0*%
_class
loc:@lstm_1/while/add_6*
out_type0*
_output_shapes
:
А
7training/Adam/gradients/lstm_1/while/add_6_grad/Shape_1Shapelstm_1/while/mul_3*
T0*%
_class
loc:@lstm_1/while/add_6*
out_type0*
_output_shapes
:
ш
Etraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2Rtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/add_6*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *%
_class
loc:@lstm_1/while/add_6*
valueB :
џџџџџџџџџ

Ktraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/Const*%
_class
loc:@lstm_1/while/add_6*

stack_name *
_output_shapes
:*
	elem_type0
Ч
Ktraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_6
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
Mtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/Const_1*

stack_name *
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/add_6
Ы
Mtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_6*
parallel_iterations 
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
StackPopV2Xtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/add_6
ю
Xtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterMtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_6

3training/Adam/gradients/lstm_1/while/add_6_grad/SumSumtraining/Adam/gradients/AddN_4Etraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_6
А
7training/Adam/gradients/lstm_1/while/add_6_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_6_grad/SumPtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/add_6*
Tshape0*'
_output_shapes
:џџџџџџџџџ

5training/Adam/gradients/lstm_1/while/add_6_grad/Sum_1Sumtraining/Adam/gradients/AddN_4Gtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs:1*
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
:џџџџџџџџџ
Ў
5training/Adam/gradients/lstm_1/while/add_8_grad/ShapeShapelstm_1/while/mul_4*
T0*%
_class
loc:@lstm_1/while/add_8*
out_type0*
_output_shapes
:
П
7training/Adam/gradients/lstm_1/while/add_8_grad/Shape_1Const^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_8*
valueB *
dtype0*
_output_shapes
: 
Э
Etraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPopV27training/Adam/gradients/lstm_1/while/add_8_grad/Shape_1*
T0*%
_class
loc:@lstm_1/while/add_8*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *%
_class
loc:@lstm_1/while/add_8*
valueB :
џџџџџџџџџ

Ktraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/add_8
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
Qtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/add_8_grad/Shape^training/Adam/gradients/Add*%
_class
loc:@lstm_1/while/add_8*
_output_shapes
:*
swap_memory(*
T0
Ј
Ptraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_8*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/f_acc*%
_class
loc:@lstm_1/while/add_8*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0
Г
3training/Adam/gradients/lstm_1/while/add_8_grad/SumSumItraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ReshapeEtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/add_8*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
А
7training/Adam/gradients/lstm_1/while/add_8_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_8_grad/SumPtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/add_8*
Tshape0*'
_output_shapes
:џџџџџџџџџ
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
9training/Adam/gradients/lstm_1/while/add_8_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_8_grad/Sum_17training/Adam/gradients/lstm_1/while/add_8_grad/Shape_1*
_output_shapes
: *
T0*%
_class
loc:@lstm_1/while/add_8*
Tshape0
И
5training/Adam/gradients/lstm_1/while/mul_2_grad/ShapeShapelstm_1/while/clip_by_value_1*
_output_shapes
:*
T0*%
_class
loc:@lstm_1/while/mul_2*
out_type0
Е
7training/Adam/gradients/lstm_1/while/mul_2_grad/Shape_1Shapelstm_1/while/Identity_4*
_output_shapes
:*
T0*%
_class
loc:@lstm_1/while/mul_2*
out_type0
ш
Etraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2Rtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/mul_2*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
Ktraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/ConstConst*
_output_shapes
: *%
_class
loc:@lstm_1/while/mul_2*
valueB :
џџџџџџџџџ*
dtype0
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
StackPopV2Vtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_2*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
П
Mtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/Const_1Const*
_output_shapes
: *%
_class
loc:@lstm_1/while/mul_2*
valueB :
џџџџџџџџџ*
dtype0

Mtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/Const_1*%
_class
loc:@lstm_1/while/mul_2*

stack_name *
_output_shapes
:*
	elem_type0
Ы
Mtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations *
is_constant(
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
:џџџџџџџџџ
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
9training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/EnterEnter9training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
Ќ
?training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPushV2StackPushV29training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/Enterlstm_1/while/Identity_4^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_2*'
_output_shapes
:џџџџџџџџџ*
swap_memory(

>training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPopV2
StackPopV2Dtraining/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_2*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
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
7training/Adam/gradients/lstm_1/while/mul_2_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/mul_2_grad/SumPtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_2*
Tshape0*'
_output_shapes
:џџџџџџџџџ

5training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1Mul@training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPopV27training/Adam/gradients/lstm_1/while/add_6_grad/Reshape*
T0*%
_class
loc:@lstm_1/while/mul_2*'
_output_shapes
:џџџџџџџџџ
а
;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/ConstConst*
_output_shapes
: *H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_2*
valueB :
џџџџџџџџџ*
dtype0
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
Atraining/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPushV2StackPushV2;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/Enterlstm_1/while/clip_by_value_1^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_2*'
_output_shapes
:џџџџџџџџџ*
swap_memory(

@training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPopV2
StackPopV2Ftraining/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPopV2/Enter^training/Adam/gradients/Sub*
	elem_type0*%
_class
loc:@lstm_1/while/mul_2*'
_output_shapes
:џџџџџџџџџ
Ъ
Ftraining/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPopV2/EnterEnter;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_2*
parallel_iterations 
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
:џџџџџџџџџ
Ж
5training/Adam/gradients/lstm_1/while/mul_3_grad/ShapeShapelstm_1/while/clip_by_value*
T0*%
_class
loc:@lstm_1/while/mul_3*
out_type0*
_output_shapes
:
Џ
7training/Adam/gradients/lstm_1/while/mul_3_grad/Shape_1Shapelstm_1/while/Tanh*
T0*%
_class
loc:@lstm_1/while/mul_3*
out_type0*
_output_shapes
:
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
Qtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/mul_3_grad/Shape^training/Adam/gradients/Add*%
_class
loc:@lstm_1/while/mul_3*
_output_shapes
:*
swap_memory(*
T0
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
Mtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *%
_class
loc:@lstm_1/while/mul_3*
valueB :
џџџџџџџџџ

Mtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Const_1*%
_class
loc:@lstm_1/while/mul_3*

stack_name *
_output_shapes
:*
	elem_type0
Ы
Mtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations *
is_constant(
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
Xtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterMtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc_1*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0

3training/Adam/gradients/lstm_1/while/mul_3_grad/MulMul9training/Adam/gradients/lstm_1/while/add_6_grad/Reshape_1>training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_3*'
_output_shapes
:џџџџџџџџџ
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
9training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/f_accStackV29training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/Const*
_output_shapes
:*
	elem_type0*=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_3*

stack_name 
Ѓ
9training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/EnterEnter9training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations 
І
?training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPushV2StackPushV29training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/Enterlstm_1/while/Tanh^training/Adam/gradients/Add*'
_output_shapes
:џџџџџџџџџ*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/mul_3

>training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPopV2
StackPopV2Dtraining/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_3*'
_output_shapes
:џџџџџџџџџ*
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
7training/Adam/gradients/lstm_1/while/mul_3_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/mul_3_grad/SumPtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/mul_3*
Tshape0

5training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1Mul@training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPopV29training/Adam/gradients/lstm_1/while/add_6_grad/Reshape_1*
T0*%
_class
loc:@lstm_1/while/mul_3*'
_output_shapes
:џџџџџџџџџ
Ю
;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/ConstConst*F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_3*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/f_accStackV2;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_3
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
Atraining/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPushV2StackPushV2;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/Enterlstm_1/while/clip_by_value^training/Adam/gradients/Add*'
_output_shapes
:џџџџџџџџџ*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/mul_3

@training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPopV2
StackPopV2Ftraining/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_3*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
Ъ
Ftraining/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPopV2/EnterEnter;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/f_acc*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_3*
parallel_iterations *
is_constant(
Ѓ
5training/Adam/gradients/lstm_1/while/mul_3_grad/Sum_1Sum5training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1Gtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/mul_3*
_output_shapes
:
Ж
9training/Adam/gradients/lstm_1/while/mul_3_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/mul_3_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopV2_1*%
_class
loc:@lstm_1/while/mul_3*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
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
Qtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/Enter7training/Adam/gradients/lstm_1/while/mul_4_grad/Shape_1^training/Adam/gradients/Add*
_output_shapes
:*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/mul_4
Ј
Ptraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/mul_4
ъ
Vtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_4*
parallel_iterations *
is_constant(

3training/Adam/gradients/lstm_1/while/mul_4_grad/MulMul7training/Adam/gradients/lstm_1/while/add_8_grad/Reshape>training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPopV2*
T0*%
_class
loc:@lstm_1/while/mul_4*'
_output_shapes
:џџџџџџџџџ
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
9training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/f_accStackV29training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/Const*
_output_shapes
:*
	elem_type0*>
_class4
2loc:@lstm_1/while/add_7loc:@lstm_1/while/mul_4*

stack_name 
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
?training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPushV2StackPushV29training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/Enterlstm_1/while/add_7^training/Adam/gradients/Add*'
_output_shapes
:џџџџџџџџџ*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/mul_4

>training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPopV2
StackPopV2Dtraining/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/mul_4*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
Ц
Dtraining/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPopV2/EnterEnter9training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/f_acc*
T0*%
_class
loc:@lstm_1/while/mul_4*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
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
7training/Adam/gradients/lstm_1/while/mul_4_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/mul_4_grad/Sum5training/Adam/gradients/lstm_1/while/mul_4_grad/Shape*
_output_shapes
: *
T0*%
_class
loc:@lstm_1/while/mul_4*
Tshape0

5training/Adam/gradients/lstm_1/while/mul_4_grad/Mul_1Mul;training/Adam/gradients/lstm_1/while/mul_4_grad/Mul_1/Const7training/Adam/gradients/lstm_1/while/add_8_grad/Reshape*
T0*%
_class
loc:@lstm_1/while/mul_4*'
_output_shapes
:џџџџџџџџџ
Х
;training/Adam/gradients/lstm_1/while/mul_4_grad/Mul_1/ConstConst^training/Adam/gradients/Sub*
dtype0*
_output_shapes
: *%
_class
loc:@lstm_1/while/mul_4*
valueB
 *ЭЬL>
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
9training/Adam/gradients/lstm_1/while/mul_4_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/mul_4_grad/Sum_1Ptraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/mul_4*
Tshape0
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
Atraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Shape_2Shape7training/Adam/gradients/lstm_1/while/mul_2_grad/Reshape*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
out_type0*
_output_shapes
:*
T0
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
?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/zerosFillAtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Shape_2Etraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/zeros/Const*'
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*

index_type0
м
Ftraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqualGreaterEqualQtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopV2Ntraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/Const_1*'
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1
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
Ltraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/EnterEnterLtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
parallel_iterations *
is_constant(
щ
Rtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPushV2StackPushV2Ltraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/Enter$lstm_1/while/clip_by_value_1/Minimum^training/Adam/gradients/Add*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*'
_output_shapes
:џџџџџџџџџ*
swap_memory(
С
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopV2
StackPopV2Wtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopV2/Enter^training/Adam/gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
і
Wtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopV2/EnterEnterLtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
parallel_iterations 
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
Otraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgsBroadcastGradientArgsZtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopV2Atraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Shape_1*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
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
[training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Utraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/Enter?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Shape^training/Adam/gradients/Add*
_output_shapes
:*
swap_memory(*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1
Ц
Ztraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2`training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
_output_shapes
:*
	elem_type0

`training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterUtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
parallel_iterations 
я
@training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/SelectSelectFtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual7training/Adam/gradients/lstm_1/while/mul_2_grad/Reshape?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/zeros*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*'
_output_shapes
:џџџџџџџџџ*
T0
ё
Btraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Select_1SelectFtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/zeros7training/Adam/gradients/lstm_1/while/mul_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1
Ш
=training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/SumSum@training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/SelectOtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
и
Atraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/ReshapeReshape=training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/SumZtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopV2*
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
Tshape0*'
_output_shapes
:џџџџџџџџџ
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
Ctraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Reshape_1Reshape?training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Sum_1Atraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Shape_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
Tshape0*
_output_shapes
: *
T0
Ю
=training/Adam/gradients/lstm_1/while/clip_by_value_grad/ShapeShape"lstm_1/while/clip_by_value/Minimum*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
out_type0*
_output_shapes
:
Я
?training/Adam/gradients/lstm_1/while/clip_by_value_grad/Shape_1Const^training/Adam/gradients/Sub*
dtype0*
_output_shapes
: *-
_class#
!loc:@lstm_1/while/clip_by_value*
valueB 
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
:џџџџџџџџџ
д
Dtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqualGreaterEqualOtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopV2Ltraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/Const_1*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*'
_output_shapes
:џџџџџџџџџ
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
Ptraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPushV2StackPushV2Jtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/Enter"lstm_1/while/clip_by_value/Minimum^training/Adam/gradients/Add*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*'
_output_shapes
:џџџџџџџџџ*
swap_memory(
Л
Otraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopV2
StackPopV2Utraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopV2/Enter^training/Adam/gradients/Sub*-
_class#
!loc:@lstm_1/while/clip_by_value*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
№
Utraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopV2/EnterEnterJtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc*-
_class#
!loc:@lstm_1/while/clip_by_value*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0
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
Straining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_accStackV2Straining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/Const*-
_class#
!loc:@lstm_1/while/clip_by_value*

stack_name *
_output_shapes
:*
	elem_type0
п
Straining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/EnterEnterStraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
parallel_iterations 
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
StackPopV2^training/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*-
_class#
!loc:@lstm_1/while/clip_by_value

^training/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopV2/EnterEnterStraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_acc*-
_class#
!loc:@lstm_1/while/clip_by_value*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0
ч
>training/Adam/gradients/lstm_1/while/clip_by_value_grad/SelectSelectDtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual7training/Adam/gradients/lstm_1/while/mul_3_grad/Reshape=training/Adam/gradients/lstm_1/while/clip_by_value_grad/zeros*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*'
_output_shapes
:џџџџџџџџџ
щ
@training/Adam/gradients/lstm_1/while/clip_by_value_grad/Select_1SelectDtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual=training/Adam/gradients/lstm_1/while/clip_by_value_grad/zeros7training/Adam/gradients/lstm_1/while/mul_3_grad/Reshape*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*'
_output_shapes
:џџџџџџџџџ
Р
;training/Adam/gradients/lstm_1/while/clip_by_value_grad/SumSum>training/Adam/gradients/lstm_1/while/clip_by_value_grad/SelectMtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
_output_shapes
:*
	keep_dims( *

Tidx0
а
?training/Adam/gradients/lstm_1/while/clip_by_value_grad/ReshapeReshape;training/Adam/gradients/lstm_1/while/clip_by_value_grad/SumXtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
Tshape0
Ц
=training/Adam/gradients/lstm_1/while/clip_by_value_grad/Sum_1Sum@training/Adam/gradients/lstm_1/while/clip_by_value_grad/Select_1Otraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs:1*
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
_output_shapes
:*
	keep_dims( *

Tidx0
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
:џџџџџџџџџ*
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
Ktraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *%
_class
loc:@lstm_1/while/add_7*
valueB :
џџџџџџџџџ
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
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/add_7
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
Mtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/Const_1*%
_class
loc:@lstm_1/while/add_7*

stack_name *
_output_shapes
:*
	elem_type0
Ы
Mtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc_1*
T0*%
_class
loc:@lstm_1/while/add_7*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
ч
Straining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/add_7_grad/Shape_1^training/Adam/gradients/Add*%
_class
loc:@lstm_1/while/add_7*
_output_shapes
:*
swap_memory(*
T0
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
:џџџџџџџџџ
Ї
5training/Adam/gradients/lstm_1/while/add_7_grad/Sum_1Sum9training/Adam/gradients/lstm_1/while/mul_4_grad/Reshape_1Gtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_7
Ж
9training/Adam/gradients/lstm_1/while/add_7_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_7_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopV2_1*
T0*%
_class
loc:@lstm_1/while/add_7*
Tshape0*'
_output_shapes
:џџџџџџџџџ
в
Gtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/ShapeShapelstm_1/while/add_4*
_output_shapes
:*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
out_type0
у
Itraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_1Const^training/Adam/gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
valueB *
dtype0*
_output_shapes
: 
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
Gtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zerosFillItraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_2Mtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros/Const*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*

index_type0*'
_output_shapes
:џџџџџџџџџ
№
Ktraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual	LessEqualVtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopV2Straining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/Const_1*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*'
_output_shapes
:џџџџџџџџџ
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
Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_accStackV2Qtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/Const*
	elem_type0*P
_classF
Dloc:@lstm_1/while/add_4)loc:@lstm_1/while/clip_by_value_1/Minimum*

stack_name *
_output_shapes
:
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
:џџџџџџџџџ*
swap_memory(
г
Vtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopV2
StackPopV2\training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopV2/Enter^training/Adam/gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
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
]training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/ConstConst*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
П
]training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_accStackV2]training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/Const*
_output_shapes
:*
	elem_type0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*

stack_name 
§
]training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/EnterEnter]training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_acc*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0
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
StackPopV2htraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
	elem_type0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
_output_shapes
:
 
htraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopV2/EnterEnter]training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
parallel_iterations 

Htraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SelectSelectKtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqualAtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/ReshapeGtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros*'
_output_shapes
:џџџџџџџџџ*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum

Jtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Select_1SelectKtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqualGtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zerosAtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Reshape*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*'
_output_shapes
:џџџџџџџџџ*
T0
ш
Etraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SumSumHtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SelectWtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
_output_shapes
:*
	keep_dims( *

Tidx0
ј
Itraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/ReshapeReshapeEtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Sumbtraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
Tshape0
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
:џџџџџџџџџ
Ю
Etraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/ShapeShapelstm_1/while/add_2*
_output_shapes
:*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
out_type0
п
Gtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_1Const^training/Adam/gradients/Sub*
dtype0*
_output_shapes
: *5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
valueB 
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
:џџџџџџџџџ
ш
Itraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual	LessEqualTtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopV2Qtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/Const_1*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ
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
:џџџџџџџџџ*
swap_memory(
Э
Ttraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopV2
StackPopV2Ztraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopV2/Enter^training/Adam/gradients/Sub*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ*
	elem_type0
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
Qtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/Const_1Const^training/Adam/gradients/Sub*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
valueB
 *  ?*
dtype0*
_output_shapes
: 
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
[training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_accStackV2[training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum
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
ftraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopV2/EnterEnter[training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_acc*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0

Ftraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/SelectSelectItraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual?training/Adam/gradients/lstm_1/while/clip_by_value_grad/ReshapeEtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros*'
_output_shapes
:џџџџџџџџџ*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum

Htraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Select_1SelectItraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqualEtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros?training/Adam/gradients/lstm_1/while/clip_by_value_grad/Reshape*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ
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
Gtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/ReshapeReshapeCtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Sum`training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopV2*
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
Tshape0*'
_output_shapes
:џџџџџџџџџ
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
Ktraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/ConstConst*
_output_shapes
: *%
_class
loc:@lstm_1/while/add_5*
valueB :
џџџџџџџџџ*
dtype0

Ktraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_accStackV2Ktraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/Const*
	elem_type0*%
_class
loc:@lstm_1/while/add_5*

stack_name *
_output_shapes
:
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
Mtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/Const_1*%
_class
loc:@lstm_1/while/add_5*

stack_name *
_output_shapes
:*
	elem_type0
Ы
Mtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_5
ч
Straining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/add_5_grad/Shape_1^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/add_5*
_output_shapes
:*
swap_memory(
Ќ
Rtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_5*
_output_shapes
:*
	elem_type0
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
:џџџџџџџџџ
Ѕ
5training/Adam/gradients/lstm_1/while/add_5_grad/Sum_1Sum7training/Adam/gradients/lstm_1/while/Tanh_grad/TanhGradGtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/add_5*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ж
9training/Adam/gradients/lstm_1/while/add_5_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_5_grad/Sum_1Rtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopV2_1*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/add_5*
Tshape0
ю
?training/Adam/gradients/lstm_1/while/BiasAdd_3_grad/BiasAddGradBiasAddGrad7training/Adam/gradients/lstm_1/while/add_7_grad/Reshape*)
_class
loc:@lstm_1/while/BiasAdd_3*
data_formatNHWC*
_output_shapes
:*
T0
С
9training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMulMatMul9training/Adam/gradients/lstm_1/while/add_7_grad/Reshape_1?training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul/Enter*
transpose_b(*
T0*(
_class
loc:@lstm_1/while/MatMul_7*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul/EnterEnterlstm_1/strided_slice_7*
is_constant(*
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_7*
parallel_iterations 
С
;training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV29training/Adam/gradients/lstm_1/while/add_7_grad/Reshape_1*
transpose_b( *
T0*(
_class
loc:@lstm_1/while/MatMul_7*
_output_shapes

:*
transpose_a(
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
Atraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/f_accStackV2Atraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/Const*
_output_shapes
:*
	elem_type0*F
_class<
:loc:@lstm_1/while/Identity_3loc:@lstm_1/while/MatMul_7*

stack_name 
Ж
Atraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/EnterEnterAtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_7*
parallel_iterations 
П
Gtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPushV2StackPushV2Atraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/Enterlstm_1/while/Identity_3^training/Adam/gradients/Add*
T0*(
_class
loc:@lstm_1/while/MatMul_7*'
_output_shapes
:џџџџџџџџџ*
swap_memory(
Є
Ftraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV2
StackPopV2Ltraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV2/Enter^training/Adam/gradients/Sub*'
_output_shapes
:џџџџџџџџџ*
	elem_type0*(
_class
loc:@lstm_1/while/MatMul_7
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
Etraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopV27training/Adam/gradients/lstm_1/while/add_4_grad/Shape_1*%
_class
loc:@lstm_1/while/add_4*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
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
Ktraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_4
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
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/add_4
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
3training/Adam/gradients/lstm_1/while/add_4_grad/SumSumItraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/ReshapeEtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@lstm_1/while/add_4
А
7training/Adam/gradients/lstm_1/while/add_4_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_4_grad/SumPtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/add_4*
Tshape0*'
_output_shapes
:џџџџџџџџџ
З
5training/Adam/gradients/lstm_1/while/add_4_grad/Sum_1SumItraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/ReshapeGtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/add_4*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
7training/Adam/gradients/lstm_1/while/add_2_grad/Shape_1Const^training/Adam/gradients/Sub*
dtype0*
_output_shapes
: *%
_class
loc:@lstm_1/while/add_2*
valueB 
Э
Etraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopV27training/Adam/gradients/lstm_1/while/add_2_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/add_2
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
Vtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_2
Б
3training/Adam/gradients/lstm_1/while/add_2_grad/SumSumGtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/ReshapeEtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs*
T0*%
_class
loc:@lstm_1/while/add_2*
_output_shapes
:*
	keep_dims( *

Tidx0
А
7training/Adam/gradients/lstm_1/while/add_2_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_2_grad/SumPtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/add_2*
Tshape0*'
_output_shapes
:џџџџџџџџџ
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
9training/Adam/gradients/lstm_1/while/add_2_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/add_2_grad/Sum_17training/Adam/gradients/lstm_1/while/add_2_grad/Shape_1*
_output_shapes
: *
T0*%
_class
loc:@lstm_1/while/add_2*
Tshape0
ю
?training/Adam/gradients/lstm_1/while/BiasAdd_2_grad/BiasAddGradBiasAddGrad7training/Adam/gradients/lstm_1/while/add_5_grad/Reshape*
data_formatNHWC*
_output_shapes
:*
T0*)
_class
loc:@lstm_1/while/BiasAdd_2
С
9training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMulMatMul9training/Adam/gradients/lstm_1/while/add_5_grad/Reshape_1?training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMul/Enter*
T0*(
_class
loc:@lstm_1/while/MatMul_6*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMul/EnterEnterlstm_1/strided_slice_6*(
_class
loc:@lstm_1/while/MatMul_6*
parallel_iterations *
is_constant(*
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0
С
;training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV29training/Adam/gradients/lstm_1/while/add_5_grad/Reshape_1*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0*(
_class
loc:@lstm_1/while/MatMul_6
П
9training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMulMatMul7training/Adam/gradients/lstm_1/while/add_7_grad/Reshape?training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul/Enter*
T0*(
_class
loc:@lstm_1/while/MatMul_3*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul/EnterEnterlstm_1/strided_slice_3*
T0*(
_class
loc:@lstm_1/while/MatMul_3*
parallel_iterations *
is_constant(*
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
П
;training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV27training/Adam/gradients/lstm_1/while/add_7_grad/Reshape*
T0*(
_class
loc:@lstm_1/while/MatMul_3*
_output_shapes

:*
transpose_a(*
transpose_b( 
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
Atraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_accStackV2Atraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/Const*M
_classC
Aloc:@lstm_1/while/MatMul_3#loc:@lstm_1/while/TensorArrayReadV3*

stack_name *
_output_shapes
:*
	elem_type0
Ж
Atraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/EnterEnterAtraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_acc*
T0*(
_class
loc:@lstm_1/while/MatMul_3*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context
Ц
Gtraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPushV2StackPushV2Atraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/Enterlstm_1/while/TensorArrayReadV3^training/Adam/gradients/Add*
T0*(
_class
loc:@lstm_1/while/MatMul_3*'
_output_shapes
:џџџџџџџџџ*
swap_memory(
Є
Ftraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV2
StackPopV2Ltraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV2/Enter^training/Adam/gradients/Sub*(
_class
loc:@lstm_1/while/MatMul_3*'
_output_shapes
:џџџџџџџџџ*
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
?training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_accConst*
dtype0*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter*
valueB*    
г
Atraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_1Enter?training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc*
parallel_iterations *
is_constant( *
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter
З
Atraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_2MergeAtraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_1Gtraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/NextIteration*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter*
N*
_output_shapes

:: 

@training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/SwitchSwitchAtraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter* 
_output_shapes
::*
T0

=training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/AddAddBtraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/Switch:1?training/Adam/gradients/lstm_1/while/BiasAdd_3_grad/BiasAddGrad*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter*
_output_shapes
:
э
Gtraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/NextIterationNextIteration=training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/Add*
_output_shapes
:*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter
с
Atraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_3Exit@training/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/Switch*
_output_shapes
:*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_3/Enter
У
>training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_accConst*
dtype0*
_output_shapes

:*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*
valueB*    
д
@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc*
is_constant( *
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*
parallel_iterations 
З
@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/NextIteration*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*
N* 
_output_shapes
:: *
T0

?training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*(
_output_shapes
::*
T0

<training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1*
T0*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*
_output_shapes

:
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/Add*
T0*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter*
_output_shapes

:
т
@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/Switch*
_output_shapes

:*
T0*.
_class$
" loc:@lstm_1/while/MatMul_7/Enter
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
Ktraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/EnterEnterKtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_1*
parallel_iterations 
у
Qtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/Enter7training/Adam/gradients/lstm_1/while/mul_1_grad/Shape_1^training/Adam/gradients/Add*
_output_shapes
:*
swap_memory(*
T0*%
_class
loc:@lstm_1/while/mul_1
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
3training/Adam/gradients/lstm_1/while/mul_1_grad/MulMul7training/Adam/gradients/lstm_1/while/add_4_grad/Reshape>training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@lstm_1/while/mul_1
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
9training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/f_accStackV29training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/Const*
_output_shapes
:*
	elem_type0*>
_class4
2loc:@lstm_1/while/add_3loc:@lstm_1/while/mul_1*

stack_name 
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
?training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPushV2StackPushV29training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/Enterlstm_1/while/add_3^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/mul_1*'
_output_shapes
:џџџџџџџџџ*
swap_memory(

>training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPopV2
StackPopV2Dtraining/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*
	elem_type0*%
_class
loc:@lstm_1/while/mul_1*'
_output_shapes
:џџџџџџџџџ
Ц
Dtraining/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPopV2/EnterEnter9training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/mul_1*
parallel_iterations 

3training/Adam/gradients/lstm_1/while/mul_1_grad/SumSum3training/Adam/gradients/lstm_1/while/mul_1_grad/MulEtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

7training/Adam/gradients/lstm_1/while/mul_1_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/mul_1_grad/Sum5training/Adam/gradients/lstm_1/while/mul_1_grad/Shape*
_output_shapes
: *
T0*%
_class
loc:@lstm_1/while/mul_1*
Tshape0

5training/Adam/gradients/lstm_1/while/mul_1_grad/Mul_1Mul;training/Adam/gradients/lstm_1/while/mul_1_grad/Mul_1/Const7training/Adam/gradients/lstm_1/while/add_4_grad/Reshape*
T0*%
_class
loc:@lstm_1/while/mul_1*'
_output_shapes
:џџџџџџџџџ
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
5training/Adam/gradients/lstm_1/while/mul_1_grad/Sum_1Sum5training/Adam/gradients/lstm_1/while/mul_1_grad/Mul_1Gtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@lstm_1/while/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
Д
9training/Adam/gradients/lstm_1/while/mul_1_grad/Reshape_1Reshape5training/Adam/gradients/lstm_1/while/mul_1_grad/Sum_1Ptraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:џџџџџџџџџ*
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
Ctraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgsBroadcastGradientArgs3training/Adam/gradients/lstm_1/while/mul_grad/ShapeNtraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopV2*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*#
_class
loc:@lstm_1/while/mul
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
Otraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Itraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/mul_grad/Shape_1^training/Adam/gradients/Add*#
_class
loc:@lstm_1/while/mul*
_output_shapes
:*
swap_memory(*
T0
Ђ
Ntraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ttraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*#
_class
loc:@lstm_1/while/mul
ф
Ttraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterItraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_acc*
T0*#
_class
loc:@lstm_1/while/mul*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context

1training/Adam/gradients/lstm_1/while/mul_grad/MulMul7training/Adam/gradients/lstm_1/while/add_2_grad/Reshape<training/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPopV2*
T0*#
_class
loc:@lstm_1/while/mul*'
_output_shapes
:џџџџџџџџџ
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
7training/Adam/gradients/lstm_1/while/mul_grad/Mul/EnterEnter7training/Adam/gradients/lstm_1/while/mul_grad/Mul/f_acc*#
_class
loc:@lstm_1/while/mul*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0
Ё
=training/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPushV2StackPushV27training/Adam/gradients/lstm_1/while/mul_grad/Mul/Enterlstm_1/while/add_1^training/Adam/gradients/Add*
T0*#
_class
loc:@lstm_1/while/mul*'
_output_shapes
:џџџџџџџџџ*
swap_memory(

<training/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPopV2
StackPopV2Btraining/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPopV2/Enter^training/Adam/gradients/Sub*#
_class
loc:@lstm_1/while/mul*'
_output_shapes
:џџџџџџџџџ*
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
1training/Adam/gradients/lstm_1/while/mul_grad/SumSum1training/Adam/gradients/lstm_1/while/mul_grad/MulCtraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*#
_class
loc:@lstm_1/while/mul*
_output_shapes
:
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
:џџџџџџџџџ
С
9training/Adam/gradients/lstm_1/while/mul_grad/Mul_1/ConstConst^training/Adam/gradients/Sub*
_output_shapes
: *#
_class
loc:@lstm_1/while/mul*
valueB
 *ЭЬL>*
dtype0
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
:џџџџџџџџџ
П
9training/Adam/gradients/lstm_1/while/MatMul_2_grad/MatMulMatMul7training/Adam/gradients/lstm_1/while/add_5_grad/Reshape?training/Adam/gradients/lstm_1/while/MatMul_2_grad/MatMul/Enter*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0*(
_class
loc:@lstm_1/while/MatMul_2
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_2_grad/MatMul/EnterEnterlstm_1/strided_slice_2*
is_constant(*
_output_shapes

:*B

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

:*
transpose_a(*
transpose_b( 
Н
?training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_accConst*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter*
valueB*    *
dtype0
г
Atraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc_1Enter?training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc*
is_constant( *
_output_shapes
:*B

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

:: 

@training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/SwitchSwitchAtraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter* 
_output_shapes
::

=training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/AddAddBtraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/Switch:1?training/Adam/gradients/lstm_1/while/BiasAdd_2_grad/BiasAddGrad*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter*
_output_shapes
:
э
Gtraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/NextIterationNextIteration=training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/Add*
_output_shapes
:*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter
с
Atraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc_3Exit@training/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/Switch*
_output_shapes
:*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_2/Enter
У
>training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_accConst*
dtype0*
_output_shapes

:*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*
valueB*    
д
@training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc*
is_constant( *
_output_shapes

:*B

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
:: 

?training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*(
_output_shapes
::*
T0

<training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMul_1*
T0*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*
_output_shapes

:
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/Add*
T0*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter*
_output_shapes

:
т
@training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/Switch*
_output_shapes

:*
T0*.
_class$
" loc:@lstm_1/while/MatMul_6/Enter
У
>training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_accConst*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
valueB*    *
dtype0*
_output_shapes

:
д
@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc*
is_constant( *
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
parallel_iterations 
З
@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/NextIteration*
N* 
_output_shapes
:: *
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter

?training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*(
_output_shapes
::

<training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1*
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
_output_shapes

:
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/Add*
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
_output_shapes

:
т
@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/Switch*
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
_output_shapes

:
А
:training/Adam/gradients/lstm_1/strided_slice_11_grad/ShapeConst**
_class 
loc:@lstm_1/strided_slice_11*
valueB: *
dtype0*
_output_shapes
:
ћ
Etraining/Adam/gradients/lstm_1/strided_slice_11_grad/StridedSliceGradStridedSliceGrad:training/Adam/gradients/lstm_1/strided_slice_11_grad/Shapelstm_1/strided_slice_11/stacklstm_1/strided_slice_11/stack_1lstm_1/strided_slice_11/stack_2Atraining/Adam/gradients/lstm_1/while/BiasAdd_3/Enter_grad/b_acc_3*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes
: *
T0*
Index0**
_class 
loc:@lstm_1/strided_slice_11
Е
9training/Adam/gradients/lstm_1/strided_slice_7_grad/ShapeConst*
_output_shapes
:*)
_class
loc:@lstm_1/strided_slice_7*
valueB"       *
dtype0
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_7_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_7_grad/Shapelstm_1/strided_slice_7/stacklstm_1/strided_slice_7/stack_1lstm_1/strided_slice_7/stack_2@training/Adam/gradients/lstm_1/while/MatMul_7/Enter_grad/b_acc_3*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

: *
Index0*
T0*)
_class
loc:@lstm_1/strided_slice_7*
shrink_axis_mask 
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
Vtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc*
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_3*
parallel_iterations 
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
Mtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/Const_1*
	elem_type0*%
_class
loc:@lstm_1/while/add_3*

stack_name *
_output_shapes
:
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
StackPopV2Xtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*
_output_shapes
:*
	elem_type0*%
_class
loc:@lstm_1/while/add_3
ю
Xtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterMtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc_1*
T0*%
_class
loc:@lstm_1/while/add_3*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
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
7training/Adam/gradients/lstm_1/while/add_3_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_3_grad/SumPtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/add_3*
Tshape0*'
_output_shapes
:џџџџџџџџџ
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
:џџџџџџџџџ
А
5training/Adam/gradients/lstm_1/while/add_1_grad/ShapeShapelstm_1/while/BiasAdd*%
_class
loc:@lstm_1/while/add_1*
out_type0*
_output_shapes
:*
T0
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
Qtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ktraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Enter5training/Adam/gradients/lstm_1/while/add_1_grad/Shape^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/add_1*
_output_shapes
:*
swap_memory(
Ј
Ptraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Vtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^training/Adam/gradients/Sub*%
_class
loc:@lstm_1/while/add_1*
_output_shapes
:*
	elem_type0
ъ
Vtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterKtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_1
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
Mtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2Mtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Const_1*%
_class
loc:@lstm_1/while/add_1*

stack_name *
_output_shapes
:*
	elem_type0
Ы
Mtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Enter_1EnterMtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_1
ч
Straining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Mtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Enter_17training/Adam/gradients/lstm_1/while/add_1_grad/Shape_1^training/Adam/gradients/Add*
T0*%
_class
loc:@lstm_1/while/add_1*
_output_shapes
:*
swap_memory(
Ќ
Rtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Xtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^training/Adam/gradients/Sub*
	elem_type0*%
_class
loc:@lstm_1/while/add_1*
_output_shapes
:
ю
Xtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterMtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*%
_class
loc:@lstm_1/while/add_1*
parallel_iterations *
is_constant(
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
7training/Adam/gradients/lstm_1/while/add_1_grad/ReshapeReshape3training/Adam/gradients/lstm_1/while/add_1_grad/SumPtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*%
_class
loc:@lstm_1/while/add_1*
Tshape0*'
_output_shapes
:џџџџџџџџџ
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
:џџџџџџџџџ
У
>training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_accConst*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
valueB*    *
dtype0*
_output_shapes

:
д
@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc*
is_constant( *
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
parallel_iterations 
З
@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/NextIteration* 
_output_shapes
:: *
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
N

?training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*(
_output_shapes
::

<training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_2_grad/MatMul_1*
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
_output_shapes

:
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/Add*
_output_shapes

:*
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter
т
@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
_output_shapes

:*
T0
А
:training/Adam/gradients/lstm_1/strided_slice_10_grad/ShapeConst*
_output_shapes
:**
_class 
loc:@lstm_1/strided_slice_10*
valueB: *
dtype0
ћ
Etraining/Adam/gradients/lstm_1/strided_slice_10_grad/StridedSliceGradStridedSliceGrad:training/Adam/gradients/lstm_1/strided_slice_10_grad/Shapelstm_1/strided_slice_10/stacklstm_1/strided_slice_10/stack_1lstm_1/strided_slice_10/stack_2Atraining/Adam/gradients/lstm_1/while/BiasAdd_2/Enter_grad/b_acc_3*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0**
_class 
loc:@lstm_1/strided_slice_10*
shrink_axis_mask 
Е
9training/Adam/gradients/lstm_1/strided_slice_6_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_6*
valueB"       *
dtype0*
_output_shapes
:
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_6_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_6_grad/Shapelstm_1/strided_slice_6/stacklstm_1/strided_slice_6/stack_1lstm_1/strided_slice_6/stack_2@training/Adam/gradients/lstm_1/while/MatMul_6/Enter_grad/b_acc_3*
new_axis_mask *
end_mask*
_output_shapes

: *
T0*
Index0*)
_class
loc:@lstm_1/strided_slice_6*
shrink_axis_mask *
ellipsis_mask *

begin_mask
Е
9training/Adam/gradients/lstm_1/strided_slice_3_grad/ShapeConst*
dtype0*
_output_shapes
:*)
_class
loc:@lstm_1/strided_slice_3*
valueB"       
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_3_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_3_grad/Shapelstm_1/strided_slice_3/stacklstm_1/strided_slice_3/stack_1lstm_1/strided_slice_3/stack_2@training/Adam/gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_3*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

: *
T0*
Index0*)
_class
loc:@lstm_1/strided_slice_3*
shrink_axis_mask 
ю
?training/Adam/gradients/lstm_1/while/BiasAdd_1_grad/BiasAddGradBiasAddGrad7training/Adam/gradients/lstm_1/while/add_3_grad/Reshape*)
_class
loc:@lstm_1/while/BiasAdd_1*
data_formatNHWC*
_output_shapes
:*
T0
С
9training/Adam/gradients/lstm_1/while/MatMul_5_grad/MatMulMatMul9training/Adam/gradients/lstm_1/while/add_3_grad/Reshape_1?training/Adam/gradients/lstm_1/while/MatMul_5_grad/MatMul/Enter*
T0*(
_class
loc:@lstm_1/while/MatMul_5*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_5_grad/MatMul/EnterEnterlstm_1/strided_slice_5*
is_constant(*
_output_shapes

:*B

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

:*
transpose_a(*
transpose_b( 
ъ
=training/Adam/gradients/lstm_1/while/BiasAdd_grad/BiasAddGradBiasAddGrad7training/Adam/gradients/lstm_1/while/add_1_grad/Reshape*
T0*'
_class
loc:@lstm_1/while/BiasAdd*
data_formatNHWC*
_output_shapes
:
С
9training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMulMatMul9training/Adam/gradients/lstm_1/while/add_1_grad/Reshape_1?training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMul/Enter*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0*(
_class
loc:@lstm_1/while/MatMul_4
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMul/EnterEnterlstm_1/strided_slice_4*
is_constant(*
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*(
_class
loc:@lstm_1/while/MatMul_4*
parallel_iterations 
С
;training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPopV29training/Adam/gradients/lstm_1/while/add_1_grad/Reshape_1*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0*(
_class
loc:@lstm_1/while/MatMul_4
Е
9training/Adam/gradients/lstm_1/strided_slice_2_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_2*
valueB"       *
dtype0*
_output_shapes
:
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_2_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_2_grad/Shapelstm_1/strided_slice_2/stacklstm_1/strided_slice_2/stack_1lstm_1/strided_slice_2/stack_2@training/Adam/gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_3*
_output_shapes

: *
T0*
Index0*)
_class
loc:@lstm_1/strided_slice_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
П
9training/Adam/gradients/lstm_1/while/MatMul_1_grad/MatMulMatMul7training/Adam/gradients/lstm_1/while/add_3_grad/Reshape?training/Adam/gradients/lstm_1/while/MatMul_1_grad/MatMul/Enter*(
_class
loc:@lstm_1/while/MatMul_1*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
Ѕ
?training/Adam/gradients/lstm_1/while/MatMul_1_grad/MatMul/EnterEnterlstm_1/strided_slice_1*
is_constant(*
_output_shapes

:*B

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

:*
transpose_a(*
transpose_b( 
Н
?training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_accConst*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter*
valueB*    *
dtype0*
_output_shapes
:
г
Atraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_1Enter?training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc*
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter*
parallel_iterations *
is_constant( 
З
Atraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_2MergeAtraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_1Gtraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/NextIteration*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter*
N*
_output_shapes

:: 

@training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/SwitchSwitchAtraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2* 
_output_shapes
::*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter

=training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/AddAddBtraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/Switch:1?training/Adam/gradients/lstm_1/while/BiasAdd_1_grad/BiasAddGrad*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter*
_output_shapes
:
э
Gtraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/NextIterationNextIteration=training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/Add*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter*
_output_shapes
:
с
Atraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_3Exit@training/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/Switch*
T0*/
_class%
#!loc:@lstm_1/while/BiasAdd_1/Enter*
_output_shapes
:
У
>training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_accConst*
_output_shapes

:*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter*
valueB*    *
dtype0
д
@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc*
T0*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter*
parallel_iterations *
is_constant( *
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
З
@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/NextIteration*
T0*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter*
N* 
_output_shapes
:: 

?training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter*(
_output_shapes
::

<training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_5_grad/MatMul_1*
_output_shapes

:*
T0*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/Add*
_output_shapes

:*
T0*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter
т
@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_5/Enter*
_output_shapes

:*
T0
Й
7training/Adam/gradients/lstm_1/while/MatMul_grad/MatMulMatMul7training/Adam/gradients/lstm_1/while/add_1_grad/Reshape=training/Adam/gradients/lstm_1/while/MatMul_grad/MatMul/Enter*
transpose_b(*
T0*&
_class
loc:@lstm_1/while/MatMul*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 

=training/Adam/gradients/lstm_1/while/MatMul_grad/MatMul/EnterEnterlstm_1/strided_slice*
is_constant(*
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*&
_class
loc:@lstm_1/while/MatMul*
parallel_iterations 
Л
9training/Adam/gradients/lstm_1/while/MatMul_grad/MatMul_1MatMulFtraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopV27training/Adam/gradients/lstm_1/while/add_1_grad/Reshape*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0*&
_class
loc:@lstm_1/while/MatMul
Й
=training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_accConst*
dtype0*
_output_shapes
:*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter*
valueB*    
Э
?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_1Enter=training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc*
parallel_iterations *
is_constant( *
_output_shapes
:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter
Џ
?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_2Merge?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_1Etraining/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/NextIteration*
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter*
N*
_output_shapes

:: 

>training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/SwitchSwitch?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter* 
_output_shapes
::

;training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/AddAdd@training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/Switch:1=training/Adam/gradients/lstm_1/while/BiasAdd_grad/BiasAddGrad*
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter*
_output_shapes
:
ч
Etraining/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/NextIterationNextIteration;training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/Add*
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter*
_output_shapes
:
л
?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_3Exit>training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/Switch*
T0*-
_class#
!loc:@lstm_1/while/BiasAdd/Enter*
_output_shapes
:
ї
training/Adam/gradients/AddN_5AddN9training/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul9training/Adam/gradients/lstm_1/while/MatMul_6_grad/MatMul9training/Adam/gradients/lstm_1/while/MatMul_5_grad/MatMul9training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMul*
T0*(
_class
loc:@lstm_1/while/MatMul_7*
N*'
_output_shapes
:џџџџџџџџџ
У
>training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_accConst*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter*
valueB*    *
dtype0*
_output_shapes

:
д
@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc*
T0*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter*
parallel_iterations *
is_constant( *
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
З
@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/NextIteration*
T0*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter*
N* 
_output_shapes
:: 

?training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*
T0*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter*(
_output_shapes
::

<training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_4_grad/MatMul_1*
_output_shapes

:*
T0*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/Add*
_output_shapes

:*
T0*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter
т
@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/Switch*
_output_shapes

:*
T0*.
_class$
" loc:@lstm_1/while/MatMul_4/Enter
У
>training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_accConst*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
valueB*    *
dtype0*
_output_shapes

:
д
@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_1Enter>training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc*
is_constant( *
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context*
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
parallel_iterations 
З
@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_2Merge@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_1Ftraining/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/NextIteration*
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
N* 
_output_shapes
:: 

?training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/SwitchSwitch@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*(
_output_shapes
::*
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter

<training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/AddAddAtraining/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/Switch:1;training/Adam/gradients/lstm_1/while/MatMul_1_grad/MatMul_1*
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
_output_shapes

:
ю
Ftraining/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/NextIterationNextIteration<training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/Add*
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
_output_shapes

:
т
@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_3Exit?training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
_output_shapes

:*
T0
Ў
9training/Adam/gradients/lstm_1/strided_slice_9_grad/ShapeConst*
_output_shapes
:*)
_class
loc:@lstm_1/strided_slice_9*
valueB: *
dtype0
ѕ
Dtraining/Adam/gradients/lstm_1/strided_slice_9_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_9_grad/Shapelstm_1/strided_slice_9/stacklstm_1/strided_slice_9/stack_1lstm_1/strided_slice_9/stack_2Atraining/Adam/gradients/lstm_1/while/BiasAdd_1/Enter_grad/b_acc_3*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*)
_class
loc:@lstm_1/strided_slice_9
Е
9training/Adam/gradients/lstm_1/strided_slice_5_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_5*
valueB"       *
dtype0*
_output_shapes
:
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_5_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_5_grad/Shapelstm_1/strided_slice_5/stacklstm_1/strided_slice_5/stack_1lstm_1/strided_slice_5/stack_2@training/Adam/gradients/lstm_1/while/MatMul_5/Enter_grad/b_acc_3*
end_mask*
_output_shapes

: *
Index0*
T0*)
_class
loc:@lstm_1/strided_slice_5*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask 
П
<training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_accConst*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
valueB*    *
dtype0*
_output_shapes

:
Ю
>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_1Enter<training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc*
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
parallel_iterations *
is_constant( *
_output_shapes

:*B

frame_name42training/Adam/gradients/lstm_1/while/while_context
Џ
>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_2Merge>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_1Dtraining/Adam/gradients/lstm_1/while/MatMul/Enter_grad/NextIteration*
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
N* 
_output_shapes
:: 

=training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/SwitchSwitch>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_2!training/Adam/gradients/b_count_2*(
_output_shapes
::*
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter

:training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/AddAdd?training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/Switch:19training/Adam/gradients/lstm_1/while/MatMul_grad/MatMul_1*
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
_output_shapes

:
ш
Dtraining/Adam/gradients/lstm_1/while/MatMul/Enter_grad/NextIterationNextIteration:training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/Add*
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
_output_shapes

:
м
>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_3Exit=training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/Switch*
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
_output_shapes

:
Ў
9training/Adam/gradients/lstm_1/strided_slice_8_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_8*
valueB: *
dtype0*
_output_shapes
:
ѓ
Dtraining/Adam/gradients/lstm_1/strided_slice_8_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_8_grad/Shapelstm_1/strided_slice_8/stacklstm_1/strided_slice_8/stack_1lstm_1/strided_slice_8/stack_2?training/Adam/gradients/lstm_1/while/BiasAdd/Enter_grad/b_acc_3*
Index0*
T0*)
_class
loc:@lstm_1/strided_slice_8*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Ю
Btraining/Adam/gradients/lstm_1/while/Switch_3_grad_1/NextIterationNextIterationtraining/Adam/gradients/AddN_5*
T0*'
_class
loc:@lstm_1/while/Merge_3*'
_output_shapes
:џџџџџџџџџ
Е
9training/Adam/gradients/lstm_1/strided_slice_4_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_4*
valueB"       *
dtype0*
_output_shapes
:
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_4_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_4_grad/Shapelstm_1/strided_slice_4/stacklstm_1/strided_slice_4/stack_1lstm_1/strided_slice_4/stack_2@training/Adam/gradients/lstm_1/while/MatMul_4/Enter_grad/b_acc_3*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

: *
Index0*
T0*)
_class
loc:@lstm_1/strided_slice_4
Е
9training/Adam/gradients/lstm_1/strided_slice_1_grad/ShapeConst*)
_class
loc:@lstm_1/strided_slice_1*
valueB"       *
dtype0*
_output_shapes
:
ј
Dtraining/Adam/gradients/lstm_1/strided_slice_1_grad/StridedSliceGradStridedSliceGrad9training/Adam/gradients/lstm_1/strided_slice_1_grad/Shapelstm_1/strided_slice_1/stacklstm_1/strided_slice_1/stack_1lstm_1/strided_slice_1/stack_2@training/Adam/gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_3*
Index0*
T0*)
_class
loc:@lstm_1/strided_slice_1*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

: 
Б
7training/Adam/gradients/lstm_1/strided_slice_grad/ShapeConst*'
_class
loc:@lstm_1/strided_slice*
valueB"       *
dtype0*
_output_shapes
:
ъ
Btraining/Adam/gradients/lstm_1/strided_slice_grad/StridedSliceGradStridedSliceGrad7training/Adam/gradients/lstm_1/strided_slice_grad/Shapelstm_1/strided_slice/stacklstm_1/strided_slice/stack_1lstm_1/strided_slice/stack_2>training/Adam/gradients/lstm_1/while/MatMul/Enter_grad/b_acc_3*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

: *
Index0*
T0*'
_class
loc:@lstm_1/strided_slice*
shrink_axis_mask 

training/Adam/gradients/AddN_6AddNEtraining/Adam/gradients/lstm_1/strided_slice_11_grad/StridedSliceGradEtraining/Adam/gradients/lstm_1/strided_slice_10_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_9_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_8_grad/StridedSliceGrad*
T0**
_class 
loc:@lstm_1/strided_slice_11*
N*
_output_shapes
: 

training/Adam/gradients/AddN_7AddNDtraining/Adam/gradients/lstm_1/strided_slice_7_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_6_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_5_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_4_grad/StridedSliceGrad*
N*
_output_shapes

: *
T0*)
_class
loc:@lstm_1/strided_slice_7

training/Adam/gradients/AddN_8AddNDtraining/Adam/gradients/lstm_1/strided_slice_3_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_2_grad/StridedSliceGradDtraining/Adam/gradients/lstm_1/strided_slice_1_grad/StridedSliceGradBtraining/Adam/gradients/lstm_1/strided_slice_grad/StridedSliceGrad*)
_class
loc:@lstm_1/strided_slice_3*
N*
_output_shapes

: *
T0
_
training/Adam/AssignAdd/valueConst*
dtype0	*
_output_shapes
: *
value	B	 R
Ќ
training/Adam/AssignAdd	AssignAddAdam/iterationstraining/Adam/AssignAdd/value*
T0	*"
_class
loc:@Adam/iterations*
_output_shapes
: *
use_locking( 
p
training/Adam/CastCastAdam/iterations/read*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
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
training/Adam/PowPowAdam/beta_2/readtraining/Adam/add*
_output_shapes
: *
T0
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
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_1*
_output_shapes
: *
T0

training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const*
_output_shapes
: *
T0
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
T0*
_output_shapes
: 
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
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
T0*
_output_shapes
: 
^
training/Adam/mulMulAdam/lr/readtraining/Adam/truediv*
_output_shapes
: *
T0
h
training/Adam/zerosConst*
valueB *    *
dtype0*
_output_shapes

: 

training/Adam/Variable
VariableV2*
_output_shapes

: *
	container *
shape
: *
shared_name *
dtype0
б
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*
validate_shape(*
_output_shapes

: *
use_locking(*
T0*)
_class
loc:@training/Adam/Variable

training/Adam/Variable/readIdentitytraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
_output_shapes

: *
T0
j
training/Adam/zeros_1Const*
valueB *    *
dtype0*
_output_shapes

: 

training/Adam/Variable_1
VariableV2*
dtype0*
_output_shapes

: *
	container *
shape
: *
shared_name 
й
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*
_output_shapes

: *
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(

training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
T0*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes

: 
b
training/Adam/zeros_2Const*
dtype0*
_output_shapes
: *
valueB *    

training/Adam/Variable_2
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
е
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2

training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*
_output_shapes
: *
T0*+
_class!
loc:@training/Adam/Variable_2
v
%training/Adam/zeros_3/shape_as_tensorConst*
valueB"      *
dtype0*
_output_shapes
:
`
training/Adam/zeros_3/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_3Fill%training/Adam/zeros_3/shape_as_tensortraining/Adam/zeros_3/Const*
T0*

index_type0* 
_output_shapes
:


training/Adam/Variable_3
VariableV2*
shared_name *
dtype0* 
_output_shapes
:
*
	container *
shape:

л
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/zeros_3*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(* 
_output_shapes
:


training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*
T0*+
_class!
loc:@training/Adam/Variable_3* 
_output_shapes
:

d
training/Adam/zeros_4Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_4
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
ж
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/zeros_4*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*
T0*+
_class!
loc:@training/Adam/Variable_4*
_output_shapes	
:
v
%training/Adam/zeros_5/shape_as_tensorConst*
valueB"      *
dtype0*
_output_shapes
:
`
training/Adam/zeros_5/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_5Fill%training/Adam/zeros_5/shape_as_tensortraining/Adam/zeros_5/Const*
T0*

index_type0* 
_output_shapes
:


training/Adam/Variable_5
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
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
T0*+
_class!
loc:@training/Adam/Variable_5* 
_output_shapes
:

d
training/Adam/zeros_6Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_6
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
ж
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
T0*+
_class!
loc:@training/Adam/Variable_6*
_output_shapes	
:
l
training/Adam/zeros_7Const*
valueB	*    *
dtype0*
_output_shapes
:	

training/Adam/Variable_7
VariableV2*
shape:	*
shared_name *
dtype0*
_output_shapes
:	*
	container 
к
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/zeros_7*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
_output_shapes
:	*
use_locking(

training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*
T0*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes
:	
b
training/Adam/zeros_8Const*
_output_shapes
:*
valueB*    *
dtype0

training/Adam/Variable_8
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
е
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8

training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
_output_shapes
:*
T0
j
training/Adam/zeros_9Const*
valueB *    *
dtype0*
_output_shapes

: 

training/Adam/Variable_9
VariableV2*
_output_shapes

: *
	container *
shape
: *
shared_name *
dtype0
й
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*
_output_shapes

: *
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(

training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
T0*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes

: 
k
training/Adam/zeros_10Const*
valueB *    *
dtype0*
_output_shapes

: 

training/Adam/Variable_10
VariableV2*
dtype0*
_output_shapes

: *
	container *
shape
: *
shared_name 
н
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes

: *
use_locking(

training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*,
_class"
 loc:@training/Adam/Variable_10*
_output_shapes

: *
T0
c
training/Adam/zeros_11Const*
valueB *    *
dtype0*
_output_shapes
: 

training/Adam/Variable_11
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
й
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/zeros_11*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(

training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*
T0*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
: 
w
&training/Adam/zeros_12/shape_as_tensorConst*
valueB"      *
dtype0*
_output_shapes
:
a
training/Adam/zeros_12/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Ё
training/Adam/zeros_12Fill&training/Adam/zeros_12/shape_as_tensortraining/Adam/zeros_12/Const* 
_output_shapes
:
*
T0*

index_type0

training/Adam/Variable_12
VariableV2*
shared_name *
dtype0* 
_output_shapes
:
*
	container *
shape:

п
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
T0*,
_class"
 loc:@training/Adam/Variable_12* 
_output_shapes
:

e
training/Adam/zeros_13Const*
dtype0*
_output_shapes	
:*
valueB*    

training/Adam/Variable_13
VariableV2*
_output_shapes	
:*
	container *
shape:*
shared_name *
dtype0
к
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*
T0*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes	
:
w
&training/Adam/zeros_14/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"      
a
training/Adam/zeros_14/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ё
training/Adam/zeros_14Fill&training/Adam/zeros_14/shape_as_tensortraining/Adam/zeros_14/Const*
T0*

index_type0* 
_output_shapes
:


training/Adam/Variable_14
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
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(* 
_output_shapes
:


training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
T0*,
_class"
 loc:@training/Adam/Variable_14* 
_output_shapes
:

e
training/Adam/zeros_15Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_15
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
к
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/zeros_15*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*
T0*,
_class"
 loc:@training/Adam/Variable_15*
_output_shapes	
:
m
training/Adam/zeros_16Const*
valueB	*    *
dtype0*
_output_shapes
:	

training/Adam/Variable_16
VariableV2*
dtype0*
_output_shapes
:	*
	container *
shape:	*
shared_name 
о
 training/Adam/Variable_16/AssignAssigntraining/Adam/Variable_16training/Adam/zeros_16*
_output_shapes
:	*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(

training/Adam/Variable_16/readIdentitytraining/Adam/Variable_16*
_output_shapes
:	*
T0*,
_class"
 loc:@training/Adam/Variable_16
c
training/Adam/zeros_17Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_17
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
й
 training/Adam/Variable_17/AssignAssigntraining/Adam/Variable_17training/Adam/zeros_17*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes
:

training/Adam/Variable_17/readIdentitytraining/Adam/Variable_17*
T0*,
_class"
 loc:@training/Adam/Variable_17*
_output_shapes
:
p
&training/Adam/zeros_18/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_18/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_18Fill&training/Adam/zeros_18/shape_as_tensortraining/Adam/zeros_18/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_18
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
й
 training/Adam/Variable_18/AssignAssigntraining/Adam/Variable_18training/Adam/zeros_18*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*
_output_shapes
:

training/Adam/Variable_18/readIdentitytraining/Adam/Variable_18*,
_class"
 loc:@training/Adam/Variable_18*
_output_shapes
:*
T0
p
&training/Adam/zeros_19/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_19/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_19Fill&training/Adam/zeros_19/shape_as_tensortraining/Adam/zeros_19/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_19
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
й
 training/Adam/Variable_19/AssignAssigntraining/Adam/Variable_19training/Adam/zeros_19*
T0*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_19/readIdentitytraining/Adam/Variable_19*
T0*,
_class"
 loc:@training/Adam/Variable_19*
_output_shapes
:
p
&training/Adam/zeros_20/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_20/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_20Fill&training/Adam/zeros_20/shape_as_tensortraining/Adam/zeros_20/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_20
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
й
 training/Adam/Variable_20/AssignAssigntraining/Adam/Variable_20training/Adam/zeros_20*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20*
validate_shape(*
_output_shapes
:

training/Adam/Variable_20/readIdentitytraining/Adam/Variable_20*
T0*,
_class"
 loc:@training/Adam/Variable_20*
_output_shapes
:
p
&training/Adam/zeros_21/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_21/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_21Fill&training/Adam/zeros_21/shape_as_tensortraining/Adam/zeros_21/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_21
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
й
 training/Adam/Variable_21/AssignAssigntraining/Adam/Variable_21training/Adam/zeros_21*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_21*
validate_shape(*
_output_shapes
:

training/Adam/Variable_21/readIdentitytraining/Adam/Variable_21*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_21
p
&training/Adam/zeros_22/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_22/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_22Fill&training/Adam/zeros_22/shape_as_tensortraining/Adam/zeros_22/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_22
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
й
 training/Adam/Variable_22/AssignAssigntraining/Adam/Variable_22training/Adam/zeros_22*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_22

training/Adam/Variable_22/readIdentitytraining/Adam/Variable_22*
T0*,
_class"
 loc:@training/Adam/Variable_22*
_output_shapes
:
p
&training/Adam/zeros_23/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_23/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_23Fill&training/Adam/zeros_23/shape_as_tensortraining/Adam/zeros_23/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_23
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
й
 training/Adam/Variable_23/AssignAssigntraining/Adam/Variable_23training/Adam/zeros_23*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_23*
validate_shape(*
_output_shapes
:

training/Adam/Variable_23/readIdentitytraining/Adam/Variable_23*,
_class"
 loc:@training/Adam/Variable_23*
_output_shapes
:*
T0
p
&training/Adam/zeros_24/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_24/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_24Fill&training/Adam/zeros_24/shape_as_tensortraining/Adam/zeros_24/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_24
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
й
 training/Adam/Variable_24/AssignAssigntraining/Adam/Variable_24training/Adam/zeros_24*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_24*
validate_shape(*
_output_shapes
:

training/Adam/Variable_24/readIdentitytraining/Adam/Variable_24*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_24
p
&training/Adam/zeros_25/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_25/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_25Fill&training/Adam/zeros_25/shape_as_tensortraining/Adam/zeros_25/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_25
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
й
 training/Adam/Variable_25/AssignAssigntraining/Adam/Variable_25training/Adam/zeros_25*,
_class"
 loc:@training/Adam/Variable_25*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_25/readIdentitytraining/Adam/Variable_25*
T0*,
_class"
 loc:@training/Adam/Variable_25*
_output_shapes
:
p
&training/Adam/zeros_26/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_26/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_26Fill&training/Adam/zeros_26/shape_as_tensortraining/Adam/zeros_26/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_26
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
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
r
training/Adam/mul_1MulAdam/beta_1/readtraining/Adam/Variable/read*
T0*
_output_shapes

: 
Z
training/Adam/sub_2/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
d
training/Adam/sub_2Subtraining/Adam/sub_2/xAdam/beta_1/read*
T0*
_output_shapes
: 
x
training/Adam/mul_2Multraining/Adam/sub_2training/Adam/gradients/AddN_8*
T0*
_output_shapes

: 
m
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*
_output_shapes

: *
T0
t
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_9/read*
T0*
_output_shapes

: 
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
g
training/Adam/SquareSquaretraining/Adam/gradients/AddN_8*
T0*
_output_shapes

: 
n
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
T0*
_output_shapes

: 
m
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
T0*
_output_shapes

: 
k
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
T0*
_output_shapes

: 
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

%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_3*
_output_shapes

: *
T0

training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_2*
_output_shapes

: *
T0
d
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0*
_output_shapes

: 
Z
training/Adam/add_3/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
p
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*
T0*
_output_shapes

: 
u
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
_output_shapes

: *
T0
p
training/Adam/sub_4Sublstm_1/kernel/readtraining/Adam/truediv_1*
T0*
_output_shapes

: 
Ш
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*
_output_shapes

: 
Ю
training/Adam/Assign_1Assigntraining/Adam/Variable_9training/Adam/add_2*
validate_shape(*
_output_shapes

: *
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9
И
training/Adam/Assign_2Assignlstm_1/kerneltraining/Adam/sub_4*
T0* 
_class
loc:@lstm_1/kernel*
validate_shape(*
_output_shapes

: *
use_locking(
t
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
_output_shapes

: *
T0
Z
training/Adam/sub_5/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
d
training/Adam/sub_5Subtraining/Adam/sub_5/xAdam/beta_1/read*
_output_shapes
: *
T0
x
training/Adam/mul_7Multraining/Adam/sub_5training/Adam/gradients/AddN_7*
T0*
_output_shapes

: 
m
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
_output_shapes

: *
T0
u
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_10/read*
T0*
_output_shapes

: 
Z
training/Adam/sub_6/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_6Subtraining/Adam/sub_6/xAdam/beta_2/read*
_output_shapes
: *
T0
i
training/Adam/Square_1Squaretraining/Adam/gradients/AddN_7*
T0*
_output_shapes

: 
p
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
T0*
_output_shapes

: 
m
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
T0*
_output_shapes

: 
l
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes

: 
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

%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_5*
T0*
_output_shapes

: 

training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_4*
_output_shapes

: *
T0
d
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
_output_shapes

: *
T0
Z
training/Adam/add_6/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
p
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
_output_shapes

: *
T0
v
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
T0*
_output_shapes

: 
z
training/Adam/sub_7Sublstm_1/recurrent_kernel/readtraining/Adam/truediv_2*
_output_shapes

: *
T0
Ю
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes

: 
а
training/Adam/Assign_4Assigntraining/Adam/Variable_10training/Adam/add_5*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes

: *
use_locking(
Ь
training/Adam/Assign_5Assignlstm_1/recurrent_kerneltraining/Adam/sub_7*
use_locking(*
T0**
_class 
loc:@lstm_1/recurrent_kernel*
validate_shape(*
_output_shapes

: 
q
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*
_output_shapes
: *
T0
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
u
training/Adam/mul_12Multraining/Adam/sub_8training/Adam/gradients/AddN_6*
_output_shapes
: *
T0
k
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
T0*
_output_shapes
: 
r
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_11/read*
T0*
_output_shapes
: 
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
e
training/Adam/Square_2Squaretraining/Adam/gradients/AddN_6*
_output_shapes
: *
T0
m
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
_output_shapes
: *
T0
k
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
T0*
_output_shapes
: 
h
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*
T0*
_output_shapes
: 
Z
training/Adam/Const_6Const*
dtype0*
_output_shapes
: *
valueB
 *    
Z
training/Adam/Const_7Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_7*
T0*
_output_shapes
: 

training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_6*
T0*
_output_shapes
: 
`
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0*
_output_shapes
: 
Z
training/Adam/add_9/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
l
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y*
T0*
_output_shapes
: 
r
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
T0*
_output_shapes
: 
k
training/Adam/sub_10Sublstm_1/bias/readtraining/Adam/truediv_3*
T0*
_output_shapes
: 
Ъ
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*
_output_shapes
: *
use_locking(
Ь
training/Adam/Assign_7Assigntraining/Adam/Variable_11training/Adam/add_8*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
: 
Б
training/Adam/Assign_8Assignlstm_1/biastraining/Adam/sub_10*
use_locking(*
T0*
_class
loc:@lstm_1/bias*
validate_shape(*
_output_shapes
: 
w
training/Adam/mul_16MulAdam/beta_1/readtraining/Adam/Variable_3/read*
T0* 
_output_shapes
:

[
training/Adam/sub_11/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_11Subtraining/Adam/sub_11/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_17Multraining/Adam/sub_114training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

r
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
T0* 
_output_shapes
:

x
training/Adam/mul_18MulAdam/beta_2/readtraining/Adam/Variable_12/read*
T0* 
_output_shapes
:

[
training/Adam/sub_12/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_12Subtraining/Adam/sub_12/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_3Square4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

t
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3* 
_output_shapes
:
*
T0
r
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
T0* 
_output_shapes
:

o
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
T0* 
_output_shapes
:

Z
training/Adam/Const_8Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_9Const*
dtype0*
_output_shapes
: *
valueB
 *  

%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_9*
T0* 
_output_shapes
:


training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_8*
T0* 
_output_shapes
:

f
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
T0* 
_output_shapes
:

[
training/Adam/add_12/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
t
training/Adam/add_12Addtraining/Adam/Sqrt_4training/Adam/add_12/y*
T0* 
_output_shapes
:

y
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
T0* 
_output_shapes
:

t
training/Adam/sub_13Subdense_1/kernel/readtraining/Adam/truediv_4* 
_output_shapes
:
*
T0
б
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(* 
_output_shapes
:
*
use_locking(
д
training/Adam/Assign_10Assigntraining/Adam/Variable_12training/Adam/add_11*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(* 
_output_shapes
:

О
training/Adam/Assign_11Assigndense_1/kerneltraining/Adam/sub_13*!
_class
loc:@dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
r
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*
T0*
_output_shapes	
:
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

training/Adam/mul_22Multraining/Adam/sub_148training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
m
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
T0*
_output_shapes	
:
s
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_13/read*
T0*
_output_shapes	
:
[
training/Adam/sub_15/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_15Subtraining/Adam/sub_15/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_4Square8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
o
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
T0*
_output_shapes	
:
m
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
T0*
_output_shapes	
:
j
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
T0*
_output_shapes	
:
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

%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_11*
T0*
_output_shapes	
:

training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_10*
_output_shapes	
:*
T0
a
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
T0*
_output_shapes	
:
[
training/Adam/add_15/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
o
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y*
T0*
_output_shapes	
:
t
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
T0*
_output_shapes	
:
m
training/Adam/sub_16Subdense_1/bias/readtraining/Adam/truediv_5*
T0*
_output_shapes	
:
Э
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
_output_shapes	
:
Я
training/Adam/Assign_13Assigntraining/Adam/Variable_13training/Adam/add_14*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes	
:
Е
training/Adam/Assign_14Assigndense_1/biastraining/Adam/sub_16*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:
w
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read*
T0* 
_output_shapes
:

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

training/Adam/mul_27Multraining/Adam/sub_174training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

r
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
T0* 
_output_shapes
:

x
training/Adam/mul_28MulAdam/beta_2/readtraining/Adam/Variable_14/read* 
_output_shapes
:
*
T0
[
training/Adam/sub_18/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_18Subtraining/Adam/sub_18/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_5Square4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

t
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
T0* 
_output_shapes
:

r
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
T0* 
_output_shapes
:

o
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
T0* 
_output_shapes
:

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

%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_13* 
_output_shapes
:
*
T0

training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_12* 
_output_shapes
:
*
T0
f
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
T0* 
_output_shapes
:

[
training/Adam/add_18/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
t
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y*
T0* 
_output_shapes
:

y
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
T0* 
_output_shapes
:

t
training/Adam/sub_19Subdense_2/kernel/readtraining/Adam/truediv_6*
T0* 
_output_shapes
:

в
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5
д
training/Adam/Assign_16Assigntraining/Adam/Variable_14training/Adam/add_17*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
О
training/Adam/Assign_17Assigndense_2/kerneltraining/Adam/sub_19*!
_class
loc:@dense_2/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
r
training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*
_output_shapes	
:*
T0
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

training/Adam/mul_32Multraining/Adam/sub_208training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
m
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
_output_shapes	
:*
T0
s
training/Adam/mul_33MulAdam/beta_2/readtraining/Adam/Variable_15/read*
T0*
_output_shapes	
:
[
training/Adam/sub_21/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_21Subtraining/Adam/sub_21/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_6Square8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
o
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
T0*
_output_shapes	
:
m
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
T0*
_output_shapes	
:
j
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
_output_shapes	
:*
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

%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_15*
_output_shapes	
:*
T0

training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_14*
_output_shapes	
:*
T0
a
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
T0*
_output_shapes	
:
[
training/Adam/add_21/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
o
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*
T0*
_output_shapes	
:
t
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
_output_shapes	
:*
T0
m
training/Adam/sub_22Subdense_2/bias/readtraining/Adam/truediv_7*
_output_shapes	
:*
T0
Э
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*
_output_shapes	
:
Я
training/Adam/Assign_19Assigntraining/Adam/Variable_15training/Adam/add_20*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes	
:
Е
training/Adam/Assign_20Assigndense_2/biastraining/Adam/sub_22*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(
v
training/Adam/mul_36MulAdam/beta_1/readtraining/Adam/Variable_7/read*
T0*
_output_shapes
:	
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

training/Adam/mul_37Multraining/Adam/sub_234training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
_output_shapes
:	*
T0
q
training/Adam/add_22Addtraining/Adam/mul_36training/Adam/mul_37*
T0*
_output_shapes
:	
w
training/Adam/mul_38MulAdam/beta_2/readtraining/Adam/Variable_16/read*
_output_shapes
:	*
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

training/Adam/Square_7Square4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	
s
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7*
_output_shapes
:	*
T0
q
training/Adam/add_23Addtraining/Adam/mul_38training/Adam/mul_39*
T0*
_output_shapes
:	
n
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22*
_output_shapes
:	*
T0
[
training/Adam/Const_16Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_17Const*
dtype0*
_output_shapes
: *
valueB
 *  

%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_17*
_output_shapes
:	*
T0

training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_16*
_output_shapes
:	*
T0
e
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
T0*
_output_shapes
:	
[
training/Adam/add_24/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
s
training/Adam/add_24Addtraining/Adam/Sqrt_8training/Adam/add_24/y*
T0*
_output_shapes
:	
x
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24*
T0*
_output_shapes
:	
s
training/Adam/sub_25Subdense_3/kernel/readtraining/Adam/truediv_8*
T0*
_output_shapes
:	
б
training/Adam/Assign_21Assigntraining/Adam/Variable_7training/Adam/add_22*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
_output_shapes
:	*
use_locking(
г
training/Adam/Assign_22Assigntraining/Adam/Variable_16training/Adam/add_23*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(*
_output_shapes
:	
Н
training/Adam/Assign_23Assigndense_3/kerneltraining/Adam/sub_25*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes
:	
q
training/Adam/mul_41MulAdam/beta_1/readtraining/Adam/Variable_8/read*
T0*
_output_shapes
:
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

training/Adam/mul_42Multraining/Adam/sub_268training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
l
training/Adam/add_25Addtraining/Adam/mul_41training/Adam/mul_42*
_output_shapes
:*
T0
r
training/Adam/mul_43MulAdam/beta_2/readtraining/Adam/Variable_17/read*
_output_shapes
:*
T0
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

training/Adam/Square_8Square8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
n
training/Adam/mul_44Multraining/Adam/sub_27training/Adam/Square_8*
_output_shapes
:*
T0
l
training/Adam/add_26Addtraining/Adam/mul_43training/Adam/mul_44*
T0*
_output_shapes
:
i
training/Adam/mul_45Multraining/Adam/multraining/Adam/add_25*
T0*
_output_shapes
:
[
training/Adam/Const_18Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_19Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_9/MinimumMinimumtraining/Adam/add_26training/Adam/Const_19*
_output_shapes
:*
T0

training/Adam/clip_by_value_9Maximum%training/Adam/clip_by_value_9/Minimumtraining/Adam/Const_18*
_output_shapes
:*
T0
`
training/Adam/Sqrt_9Sqrttraining/Adam/clip_by_value_9*
T0*
_output_shapes
:
[
training/Adam/add_27/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
n
training/Adam/add_27Addtraining/Adam/Sqrt_9training/Adam/add_27/y*
T0*
_output_shapes
:
s
training/Adam/truediv_9RealDivtraining/Adam/mul_45training/Adam/add_27*
T0*
_output_shapes
:
l
training/Adam/sub_28Subdense_3/bias/readtraining/Adam/truediv_9*
T0*
_output_shapes
:
Ь
training/Adam/Assign_24Assigntraining/Adam/Variable_8training/Adam/add_25*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*
_output_shapes
:
Ю
training/Adam/Assign_25Assigntraining/Adam/Variable_17training/Adam/add_26*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes
:
Д
training/Adam/Assign_26Assigndense_3/biastraining/Adam/sub_28*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:

training/group_depsNoOp	^loss/mul^metrics/acc/Mean^training/Adam/Assign^training/Adam/AssignAdd^training/Adam/Assign_1^training/Adam/Assign_10^training/Adam/Assign_11^training/Adam/Assign_12^training/Adam/Assign_13^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_18^training/Adam/Assign_19^training/Adam/Assign_2^training/Adam/Assign_20^training/Adam/Assign_21^training/Adam/Assign_22^training/Adam/Assign_23^training/Adam/Assign_24^training/Adam/Assign_25^training/Adam/Assign_26^training/Adam/Assign_3^training/Adam/Assign_4^training/Adam/Assign_5^training/Adam/Assign_6^training/Adam/Assign_7^training/Adam/Assign_8^training/Adam/Assign_9

IsVariableInitializedIsVariableInitializedlstm_1/kernel* 
_class
loc:@lstm_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_1IsVariableInitializedlstm_1/recurrent_kernel**
_class 
loc:@lstm_1/recurrent_kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_2IsVariableInitializedlstm_1/bias*
_class
loc:@lstm_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_3IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_4IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_5IsVariableInitializeddense_2/kernel*
_output_shapes
: *!
_class
loc:@dense_2/kernel*
dtype0

IsVariableInitialized_6IsVariableInitializeddense_2/bias*
_output_shapes
: *
_class
loc:@dense_2/bias*
dtype0

IsVariableInitialized_7IsVariableInitializeddense_3/kernel*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_8IsVariableInitializeddense_3/bias*
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_9IsVariableInitializedAdam/iterations*"
_class
loc:@Adam/iterations*
dtype0	*
_output_shapes
: 
{
IsVariableInitialized_10IsVariableInitializedAdam/lr*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 

IsVariableInitialized_11IsVariableInitializedAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_12IsVariableInitializedAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 

IsVariableInitialized_13IsVariableInitialized
Adam/decay*
_output_shapes
: *
_class
loc:@Adam/decay*
dtype0

IsVariableInitialized_14IsVariableInitializedtraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
dtype0*
_output_shapes
: 

IsVariableInitialized_15IsVariableInitializedtraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_16IsVariableInitializedtraining/Adam/Variable_2*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_2

IsVariableInitialized_17IsVariableInitializedtraining/Adam/Variable_3*+
_class!
loc:@training/Adam/Variable_3*
dtype0*
_output_shapes
: 

IsVariableInitialized_18IsVariableInitializedtraining/Adam/Variable_4*+
_class!
loc:@training/Adam/Variable_4*
dtype0*
_output_shapes
: 

IsVariableInitialized_19IsVariableInitializedtraining/Adam/Variable_5*+
_class!
loc:@training/Adam/Variable_5*
dtype0*
_output_shapes
: 

IsVariableInitialized_20IsVariableInitializedtraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*
dtype0*
_output_shapes
: 

IsVariableInitialized_21IsVariableInitializedtraining/Adam/Variable_7*+
_class!
loc:@training/Adam/Variable_7*
dtype0*
_output_shapes
: 

IsVariableInitialized_22IsVariableInitializedtraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
dtype0*
_output_shapes
: 

IsVariableInitialized_23IsVariableInitializedtraining/Adam/Variable_9*+
_class!
loc:@training/Adam/Variable_9*
dtype0*
_output_shapes
: 

IsVariableInitialized_24IsVariableInitializedtraining/Adam/Variable_10*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_10

IsVariableInitialized_25IsVariableInitializedtraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
dtype0*
_output_shapes
: 

IsVariableInitialized_26IsVariableInitializedtraining/Adam/Variable_12*,
_class"
 loc:@training/Adam/Variable_12*
dtype0*
_output_shapes
: 

IsVariableInitialized_27IsVariableInitializedtraining/Adam/Variable_13*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_13*
dtype0

IsVariableInitialized_28IsVariableInitializedtraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
dtype0*
_output_shapes
: 

IsVariableInitialized_29IsVariableInitializedtraining/Adam/Variable_15*,
_class"
 loc:@training/Adam/Variable_15*
dtype0*
_output_shapes
: 

IsVariableInitialized_30IsVariableInitializedtraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*
dtype0*
_output_shapes
: 

IsVariableInitialized_31IsVariableInitializedtraining/Adam/Variable_17*,
_class"
 loc:@training/Adam/Variable_17*
dtype0*
_output_shapes
: 

IsVariableInitialized_32IsVariableInitializedtraining/Adam/Variable_18*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_18

IsVariableInitialized_33IsVariableInitializedtraining/Adam/Variable_19*,
_class"
 loc:@training/Adam/Variable_19*
dtype0*
_output_shapes
: 

IsVariableInitialized_34IsVariableInitializedtraining/Adam/Variable_20*,
_class"
 loc:@training/Adam/Variable_20*
dtype0*
_output_shapes
: 

IsVariableInitialized_35IsVariableInitializedtraining/Adam/Variable_21*,
_class"
 loc:@training/Adam/Variable_21*
dtype0*
_output_shapes
: 

IsVariableInitialized_36IsVariableInitializedtraining/Adam/Variable_22*,
_class"
 loc:@training/Adam/Variable_22*
dtype0*
_output_shapes
: 

IsVariableInitialized_37IsVariableInitializedtraining/Adam/Variable_23*,
_class"
 loc:@training/Adam/Variable_23*
dtype0*
_output_shapes
: 

IsVariableInitialized_38IsVariableInitializedtraining/Adam/Variable_24*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_24*
dtype0

IsVariableInitialized_39IsVariableInitializedtraining/Adam/Variable_25*,
_class"
 loc:@training/Adam/Variable_25*
dtype0*
_output_shapes
: 

IsVariableInitialized_40IsVariableInitializedtraining/Adam/Variable_26*,
_class"
 loc:@training/Adam/Variable_26*
dtype0*
_output_shapes
: 
№	
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^dense_3/bias/Assign^dense_3/kernel/Assign^lstm_1/bias/Assign^lstm_1/kernel/Assign^lstm_1/recurrent_kernel/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign!^training/Adam/Variable_18/Assign!^training/Adam/Variable_19/Assign ^training/Adam/Variable_2/Assign!^training/Adam/Variable_20/Assign!^training/Adam/Variable_21/Assign!^training/Adam/Variable_22/Assign!^training/Adam/Variable_23/Assign!^training/Adam/Variable_24/Assign!^training/Adam/Variable_25/Assign!^training/Adam/Variable_26/Assign ^training/Adam/Variable_3/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign""К$
trainable_variablesЂ$$
X
lstm_1/kernel:0lstm_1/kernel/Assignlstm_1/kernel/read:02lstm_1/random_uniform:08

lstm_1/recurrent_kernel:0lstm_1/recurrent_kernel/Assignlstm_1/recurrent_kernel/read:02'lstm_1/recurrent_kernel/initial_value:08
J
lstm_1/bias:0lstm_1/bias/Assignlstm_1/bias/read:02lstm_1/concat:08
\
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02dense_1/random_uniform:08
M
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02dense_1/Const:08
\
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02dense_2/random_uniform:08
M
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02dense_2/Const:08
\
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:02dense_3/random_uniform:08
M
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:02dense_3/Const:08
f
Adam/iterations:0Adam/iterations/AssignAdam/iterations/read:02Adam/iterations/initial_value:08
F
	Adam/lr:0Adam/lr/AssignAdam/lr/read:02Adam/lr/initial_value:08
V
Adam/beta_1:0Adam/beta_1/AssignAdam/beta_1/read:02Adam/beta_1/initial_value:08
V
Adam/beta_2:0Adam/beta_2/AssignAdam/beta_2/read:02Adam/beta_2/initial_value:08
R
Adam/decay:0Adam/decay/AssignAdam/decay/read:02Adam/decay/initial_value:08
q
training/Adam/Variable:0training/Adam/Variable/Assigntraining/Adam/Variable/read:02training/Adam/zeros:08
y
training/Adam/Variable_1:0training/Adam/Variable_1/Assigntraining/Adam/Variable_1/read:02training/Adam/zeros_1:08
y
training/Adam/Variable_2:0training/Adam/Variable_2/Assigntraining/Adam/Variable_2/read:02training/Adam/zeros_2:08
y
training/Adam/Variable_3:0training/Adam/Variable_3/Assigntraining/Adam/Variable_3/read:02training/Adam/zeros_3:08
y
training/Adam/Variable_4:0training/Adam/Variable_4/Assigntraining/Adam/Variable_4/read:02training/Adam/zeros_4:08
y
training/Adam/Variable_5:0training/Adam/Variable_5/Assigntraining/Adam/Variable_5/read:02training/Adam/zeros_5:08
y
training/Adam/Variable_6:0training/Adam/Variable_6/Assigntraining/Adam/Variable_6/read:02training/Adam/zeros_6:08
y
training/Adam/Variable_7:0training/Adam/Variable_7/Assigntraining/Adam/Variable_7/read:02training/Adam/zeros_7:08
y
training/Adam/Variable_8:0training/Adam/Variable_8/Assigntraining/Adam/Variable_8/read:02training/Adam/zeros_8:08
y
training/Adam/Variable_9:0training/Adam/Variable_9/Assigntraining/Adam/Variable_9/read:02training/Adam/zeros_9:08
}
training/Adam/Variable_10:0 training/Adam/Variable_10/Assign training/Adam/Variable_10/read:02training/Adam/zeros_10:08
}
training/Adam/Variable_11:0 training/Adam/Variable_11/Assign training/Adam/Variable_11/read:02training/Adam/zeros_11:08
}
training/Adam/Variable_12:0 training/Adam/Variable_12/Assign training/Adam/Variable_12/read:02training/Adam/zeros_12:08
}
training/Adam/Variable_13:0 training/Adam/Variable_13/Assign training/Adam/Variable_13/read:02training/Adam/zeros_13:08
}
training/Adam/Variable_14:0 training/Adam/Variable_14/Assign training/Adam/Variable_14/read:02training/Adam/zeros_14:08
}
training/Adam/Variable_15:0 training/Adam/Variable_15/Assign training/Adam/Variable_15/read:02training/Adam/zeros_15:08
}
training/Adam/Variable_16:0 training/Adam/Variable_16/Assign training/Adam/Variable_16/read:02training/Adam/zeros_16:08
}
training/Adam/Variable_17:0 training/Adam/Variable_17/Assign training/Adam/Variable_17/read:02training/Adam/zeros_17:08
}
training/Adam/Variable_18:0 training/Adam/Variable_18/Assign training/Adam/Variable_18/read:02training/Adam/zeros_18:08
}
training/Adam/Variable_19:0 training/Adam/Variable_19/Assign training/Adam/Variable_19/read:02training/Adam/zeros_19:08
}
training/Adam/Variable_20:0 training/Adam/Variable_20/Assign training/Adam/Variable_20/read:02training/Adam/zeros_20:08
}
training/Adam/Variable_21:0 training/Adam/Variable_21/Assign training/Adam/Variable_21/read:02training/Adam/zeros_21:08
}
training/Adam/Variable_22:0 training/Adam/Variable_22/Assign training/Adam/Variable_22/read:02training/Adam/zeros_22:08
}
training/Adam/Variable_23:0 training/Adam/Variable_23/Assign training/Adam/Variable_23/read:02training/Adam/zeros_23:08
}
training/Adam/Variable_24:0 training/Adam/Variable_24/Assign training/Adam/Variable_24/read:02training/Adam/zeros_24:08
}
training/Adam/Variable_25:0 training/Adam/Variable_25/Assign training/Adam/Variable_25/read:02training/Adam/zeros_25:08
}
training/Adam/Variable_26:0 training/Adam/Variable_26/Assign training/Adam/Variable_26/read:02training/Adam/zeros_26:08"К
cond_contextЉІ

dropout_1/cond/cond_textdropout_1/cond/pred_id:0dropout_1/cond/switch_t:0 *Ч
dense_1/Relu:0
dropout_1/cond/dropout/Floor:0
dropout_1/cond/dropout/Shape:0
dropout_1/cond/dropout/add:0
dropout_1/cond/dropout/div:0
"dropout_1/cond/dropout/keep_prob:0
dropout_1/cond/dropout/mul:0
5dropout_1/cond/dropout/random_uniform/RandomUniform:0
+dropout_1/cond/dropout/random_uniform/max:0
+dropout_1/cond/dropout/random_uniform/min:0
+dropout_1/cond/dropout/random_uniform/mul:0
+dropout_1/cond/dropout/random_uniform/sub:0
'dropout_1/cond/dropout/random_uniform:0
dropout_1/cond/mul/Switch:1
dropout_1/cond/mul/y:0
dropout_1/cond/mul:0
dropout_1/cond/pred_id:0
dropout_1/cond/switch_t:0-
dense_1/Relu:0dropout_1/cond/mul/Switch:14
dropout_1/cond/pred_id:0dropout_1/cond/pred_id:0
В
dropout_1/cond/cond_text_1dropout_1/cond/pred_id:0dropout_1/cond/switch_f:0*о
dense_1/Relu:0
dropout_1/cond/Switch_1:0
dropout_1/cond/Switch_1:1
dropout_1/cond/pred_id:0
dropout_1/cond/switch_f:0+
dense_1/Relu:0dropout_1/cond/Switch_1:04
dropout_1/cond/pred_id:0dropout_1/cond/pred_id:0

dropout_2/cond/cond_textdropout_2/cond/pred_id:0dropout_2/cond/switch_t:0 *Ч
dense_2/Relu:0
dropout_2/cond/dropout/Floor:0
dropout_2/cond/dropout/Shape:0
dropout_2/cond/dropout/add:0
dropout_2/cond/dropout/div:0
"dropout_2/cond/dropout/keep_prob:0
dropout_2/cond/dropout/mul:0
5dropout_2/cond/dropout/random_uniform/RandomUniform:0
+dropout_2/cond/dropout/random_uniform/max:0
+dropout_2/cond/dropout/random_uniform/min:0
+dropout_2/cond/dropout/random_uniform/mul:0
+dropout_2/cond/dropout/random_uniform/sub:0
'dropout_2/cond/dropout/random_uniform:0
dropout_2/cond/mul/Switch:1
dropout_2/cond/mul/y:0
dropout_2/cond/mul:0
dropout_2/cond/pred_id:0
dropout_2/cond/switch_t:0-
dense_2/Relu:0dropout_2/cond/mul/Switch:14
dropout_2/cond/pred_id:0dropout_2/cond/pred_id:0
В
dropout_2/cond/cond_text_1dropout_2/cond/pred_id:0dropout_2/cond/switch_f:0*о
dense_2/Relu:0
dropout_2/cond/Switch_1:0
dropout_2/cond/Switch_1:1
dropout_2/cond/pred_id:0
dropout_2/cond/switch_f:0+
dense_2/Relu:0dropout_2/cond/Switch_1:04
dropout_2/cond/pred_id:0dropout_2/cond/pred_id:0"ѓЦ
while_contextрЦмЦ
иЦ
lstm_1/while/while_context  *lstm_1/while/LoopCond:02lstm_1/while/Merge:0:lstm_1/while/Identity:0Blstm_1/while/Exit:0Blstm_1/while/Exit_1:0Blstm_1/while/Exit_2:0Blstm_1/while/Exit_3:0Blstm_1/while/Exit_4:0B#training/Adam/gradients/f_count_2:0JУ
lstm_1/TensorArray:0
Clstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
lstm_1/TensorArray_1:0
lstm_1/strided_slice:0
lstm_1/strided_slice_10:0
lstm_1/strided_slice_11:0
lstm_1/strided_slice_12:0
lstm_1/strided_slice_1:0
lstm_1/strided_slice_2:0
lstm_1/strided_slice_3:0
lstm_1/strided_slice_4:0
lstm_1/strided_slice_5:0
lstm_1/strided_slice_6:0
lstm_1/strided_slice_7:0
lstm_1/strided_slice_8:0
lstm_1/strided_slice_9:0
lstm_1/while/BiasAdd/Enter:0
lstm_1/while/BiasAdd:0
lstm_1/while/BiasAdd_1/Enter:0
lstm_1/while/BiasAdd_1:0
lstm_1/while/BiasAdd_2/Enter:0
lstm_1/while/BiasAdd_2:0
lstm_1/while/BiasAdd_3/Enter:0
lstm_1/while/BiasAdd_3:0
lstm_1/while/Const:0
lstm_1/while/Const_1:0
lstm_1/while/Const_2:0
lstm_1/while/Const_3:0
lstm_1/while/Const_4:0
lstm_1/while/Const_5:0
lstm_1/while/Enter:0
lstm_1/while/Enter_1:0
lstm_1/while/Enter_2:0
lstm_1/while/Enter_3:0
lstm_1/while/Enter_4:0
lstm_1/while/Exit:0
lstm_1/while/Exit_1:0
lstm_1/while/Exit_2:0
lstm_1/while/Exit_3:0
lstm_1/while/Exit_4:0
lstm_1/while/Identity:0
lstm_1/while/Identity_1:0
lstm_1/while/Identity_2:0
lstm_1/while/Identity_3:0
lstm_1/while/Identity_4:0
lstm_1/while/Less/Enter:0
lstm_1/while/Less:0
lstm_1/while/Less_1/Enter:0
lstm_1/while/Less_1:0
lstm_1/while/LogicalAnd:0
lstm_1/while/LoopCond:0
lstm_1/while/MatMul/Enter:0
lstm_1/while/MatMul:0
lstm_1/while/MatMul_1/Enter:0
lstm_1/while/MatMul_1:0
lstm_1/while/MatMul_2/Enter:0
lstm_1/while/MatMul_2:0
lstm_1/while/MatMul_3/Enter:0
lstm_1/while/MatMul_3:0
lstm_1/while/MatMul_4/Enter:0
lstm_1/while/MatMul_4:0
lstm_1/while/MatMul_5/Enter:0
lstm_1/while/MatMul_5:0
lstm_1/while/MatMul_6/Enter:0
lstm_1/while/MatMul_6:0
lstm_1/while/MatMul_7/Enter:0
lstm_1/while/MatMul_7:0
lstm_1/while/Merge:0
lstm_1/while/Merge:1
lstm_1/while/Merge_1:0
lstm_1/while/Merge_1:1
lstm_1/while/Merge_2:0
lstm_1/while/Merge_2:1
lstm_1/while/Merge_3:0
lstm_1/while/Merge_3:1
lstm_1/while/Merge_4:0
lstm_1/while/Merge_4:1
lstm_1/while/NextIteration:0
lstm_1/while/NextIteration_1:0
lstm_1/while/NextIteration_2:0
lstm_1/while/NextIteration_3:0
lstm_1/while/NextIteration_4:0
lstm_1/while/Switch:0
lstm_1/while/Switch:1
lstm_1/while/Switch_1:0
lstm_1/while/Switch_1:1
lstm_1/while/Switch_2:0
lstm_1/while/Switch_2:1
lstm_1/while/Switch_3:0
lstm_1/while/Switch_3:1
lstm_1/while/Switch_4:0
lstm_1/while/Switch_4:1
lstm_1/while/Tanh:0
lstm_1/while/Tanh_1:0
&lstm_1/while/TensorArrayReadV3/Enter:0
(lstm_1/while/TensorArrayReadV3/Enter_1:0
 lstm_1/while/TensorArrayReadV3:0
8lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
2lstm_1/while/TensorArrayWrite/TensorArrayWriteV3:0
lstm_1/while/add/y:0
lstm_1/while/add:0
lstm_1/while/add_1:0
lstm_1/while/add_2/y:0
lstm_1/while/add_2:0
lstm_1/while/add_3:0
lstm_1/while/add_4/y:0
lstm_1/while/add_4:0
lstm_1/while/add_5:0
lstm_1/while/add_6:0
lstm_1/while/add_7:0
lstm_1/while/add_8/y:0
lstm_1/while/add_8:0
lstm_1/while/add_9/y:0
lstm_1/while/add_9:0
$lstm_1/while/clip_by_value/Minimum:0
lstm_1/while/clip_by_value:0
&lstm_1/while/clip_by_value_1/Minimum:0
lstm_1/while/clip_by_value_1:0
&lstm_1/while/clip_by_value_2/Minimum:0
lstm_1/while/clip_by_value_2:0
!lstm_1/while/maximum_iterations:0
lstm_1/while/mul/x:0
lstm_1/while/mul:0
lstm_1/while/mul_1/x:0
lstm_1/while/mul_1:0
lstm_1/while/mul_2:0
lstm_1/while/mul_3:0
lstm_1/while/mul_4/x:0
lstm_1/while/mul_4:0
lstm_1/while/mul_5:0
training/Adam/gradients/Add/y:0
training/Adam/gradients/Add:0
training/Adam/gradients/Merge:0
training/Adam/gradients/Merge:1
'training/Adam/gradients/NextIteration:0
 training/Adam/gradients/Switch:0
 training/Adam/gradients/Switch:1
!training/Adam/gradients/f_count:0
#training/Adam/gradients/f_count_1:0
#training/Adam/gradients/f_count_2:0
Ctraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/Enter:0
Itraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPushV2:0
Ctraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_acc:0
Ctraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/Enter:0
Itraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/StackPushV2:0
Ctraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/f_acc:0
gtraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
mtraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
gtraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
Mtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Enter:0
Otraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Enter_1:0
Straining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPushV2:0
Utraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPushV2_1:0
Mtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc:0
Otraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc_1:0
7training/Adam/gradients/lstm_1/while/add_1_grad/Shape:0
9training/Adam/gradients/lstm_1/while/add_1_grad/Shape_1:0
Mtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/Enter:0
Straining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPushV2:0
Mtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc:0
7training/Adam/gradients/lstm_1/while/add_2_grad/Shape:0
Mtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/Enter:0
Otraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/Enter_1:0
Straining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPushV2:0
Utraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPushV2_1:0
Mtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc:0
Otraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc_1:0
7training/Adam/gradients/lstm_1/while/add_3_grad/Shape:0
9training/Adam/gradients/lstm_1/while/add_3_grad/Shape_1:0
Mtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/Enter:0
Straining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPushV2:0
Mtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc:0
7training/Adam/gradients/lstm_1/while/add_4_grad/Shape:0
Mtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/Enter:0
Otraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/Enter_1:0
Straining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPushV2:0
Utraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPushV2_1:0
Mtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc:0
Otraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1:0
7training/Adam/gradients/lstm_1/while/add_5_grad/Shape:0
9training/Adam/gradients/lstm_1/while/add_5_grad/Shape_1:0
Mtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/Enter:0
Otraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/Enter_1:0
Straining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPushV2:0
Utraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPushV2_1:0
Mtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc:0
Otraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1:0
7training/Adam/gradients/lstm_1/while/add_6_grad/Shape:0
9training/Adam/gradients/lstm_1/while/add_6_grad/Shape_1:0
Mtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/Enter:0
Otraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/Enter_1:0
Straining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPushV2:0
Utraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPushV2_1:0
Mtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc:0
Otraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc_1:0
7training/Adam/gradients/lstm_1/while/add_7_grad/Shape:0
9training/Adam/gradients/lstm_1/while/add_7_grad/Shape_1:0
Mtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/Enter:0
Straining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/StackPushV2:0
Mtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/f_acc:0
7training/Adam/gradients/lstm_1/while/add_8_grad/Shape:0
]training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/Enter:0
ctraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPushV2:0
]training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_acc:0
Qtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/Enter:0
Wtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPushV2:0
Qtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc:0
Gtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape:0
_training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/Enter:0
etraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPushV2:0
_training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_acc:0
Straining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/Enter:0
Ytraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPushV2:0
Straining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc:0
Itraining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape:0
Wtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/Enter:0
]training/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPushV2:0
Wtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_acc:0
Ntraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/Enter:0
Ttraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPushV2:0
Ntraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc:0
Atraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/Shape:0
_training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/Enter:0
etraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPushV2:0
_training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_acc:0
Straining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/Enter:0
Ytraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPushV2:0
Straining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc:0
Itraining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape:0
Wtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/Enter:0
]training/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPushV2:0
Wtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_acc:0
Ntraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/Enter:0
Ttraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPushV2:0
Ntraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc:0
Atraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/Shape:0
Utraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/Enter:0
[training/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPushV2:0
Utraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_acc:0
Ltraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/Enter:0
Rtraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPushV2:0
Ltraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc:0
?training/Adam/gradients/lstm_1/while/clip_by_value_grad/Shape:0
Mtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/Enter:0
Straining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPushV2:0
Mtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_acc:0
;training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/Enter:0
Atraining/Adam/gradients/lstm_1/while/mul_1_grad/Mul/StackPushV2:0
;training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/f_acc:0
9training/Adam/gradients/lstm_1/while/mul_1_grad/Shape_1:0
Mtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/Enter:0
Otraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/Enter_1:0
Straining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPushV2:0
Utraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPushV2_1:0
Mtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc:0
Otraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc_1:0
;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/Enter:0
Atraining/Adam/gradients/lstm_1/while/mul_2_grad/Mul/StackPushV2:0
;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/f_acc:0
=training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/Enter:0
Ctraining/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/StackPushV2:0
=training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/f_acc:0
7training/Adam/gradients/lstm_1/while/mul_2_grad/Shape:0
9training/Adam/gradients/lstm_1/while/mul_2_grad/Shape_1:0
Mtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Enter:0
Otraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Enter_1:0
Straining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPushV2:0
Utraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPushV2_1:0
Mtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc:0
Otraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc_1:0
;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/Enter:0
Atraining/Adam/gradients/lstm_1/while/mul_3_grad/Mul/StackPushV2:0
;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/f_acc:0
=training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/Enter:0
Ctraining/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/StackPushV2:0
=training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/f_acc:0
7training/Adam/gradients/lstm_1/while/mul_3_grad/Shape:0
9training/Adam/gradients/lstm_1/while/mul_3_grad/Shape_1:0
Mtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/Enter:0
Straining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPushV2:0
Mtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc:0
;training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/Enter:0
Atraining/Adam/gradients/lstm_1/while/mul_4_grad/Mul/StackPushV2:0
;training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/f_acc:0
9training/Adam/gradients/lstm_1/while/mul_4_grad/Shape_1:0
Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Enter:0
Otraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Enter_1:0
Straining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPushV2:0
Utraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPushV2_1:0
Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc:0
Otraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc_1:0
;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/Enter:0
Atraining/Adam/gradients/lstm_1/while/mul_5_grad/Mul/StackPushV2:0
;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/f_acc:0
=training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/Enter:0
Ctraining/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/StackPushV2:0
=training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/f_acc:0
7training/Adam/gradients/lstm_1/while/mul_5_grad/Shape:0
9training/Adam/gradients/lstm_1/while/mul_5_grad/Shape_1:0
Ktraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/Enter:0
Qtraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPushV2:0
Ktraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_acc:0
9training/Adam/gradients/lstm_1/while/mul_grad/Mul/Enter:0
?training/Adam/gradients/lstm_1/while/mul_grad/Mul/StackPushV2:0
9training/Adam/gradients/lstm_1/while/mul_grad/Mul/f_acc:0
7training/Adam/gradients/lstm_1/while/mul_grad/Shape_1:0Ђ
Otraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1:0Otraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/Enter_1:0z
;training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/f_acc:0;training/Adam/gradients/lstm_1/while/mul_4_grad/Mul/Enter:0
Ktraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_acc:0Ktraining/Adam/gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/Enter:0В
Wtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_acc:0Wtraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/Enter:09
lstm_1/strided_slice_6:0lstm_1/while/MatMul_6/Enter:0В
Wtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_acc:0Wtraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/Enter:09
lstm_1/strided_slice_1:0lstm_1/while/MatMul_1/Enter:0P
lstm_1/TensorArray:08lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:08
lstm_1/strided_slice_12:0lstm_1/while/Less_1/Enter:0Ђ
Otraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc_1:0Otraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Enter_1:0Ђ
Otraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc_1:0Otraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/Enter_1:0
Ctraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/f_acc:0Ctraining/Adam/gradients/lstm_1/while/MatMul_7_grad/MatMul_1/Enter:0Ў
Utraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_acc:0Utraining/Adam/gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/Enter:0~
=training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/f_acc:0=training/Adam/gradients/lstm_1/while/mul_2_grad/Mul_1/Enter:0Т
_training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_acc:0_training/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/Enter:09
lstm_1/strided_slice_7:0lstm_1/while/MatMul_7/Enter:09
lstm_1/strided_slice_2:0lstm_1/while/MatMul_2/Enter:0
Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc:0Mtraining/Adam/gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/Enter:0 
Ntraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc:0Ntraining/Adam/gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/Enter:0
Mtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc:0Mtraining/Adam/gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/Enter:0v
9training/Adam/gradients/lstm_1/while/mul_grad/Mul/f_acc:09training/Adam/gradients/lstm_1/while/mul_grad/Mul/Enter:0
Mtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc:0Mtraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Enter:0
Mtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc:0Mtraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/Enter:0Ђ
Otraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc_1:0Otraining/Adam/gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/Enter_1:0Ђ
Otraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc_1:0Otraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Enter_1:0
Mtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_acc:0Mtraining/Adam/gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/Enter:0О
]training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_acc:0]training/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/Enter:0z
;training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/f_acc:0;training/Adam/gradients/lstm_1/while/mul_1_grad/Mul/Enter:0Ђ
Otraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1:0Otraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/Enter_1:08
lstm_1/strided_slice_8:0lstm_1/while/BiasAdd/Enter:09
lstm_1/strided_slice_3:0lstm_1/while/MatMul_3/Enter:0Њ
Straining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc:0Straining/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/Enter:0
Mtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/f_acc:0Mtraining/Adam/gradients/lstm_1/while/add_8_grad/BroadcastGradientArgs/Enter:0z
;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/f_acc:0;training/Adam/gradients/lstm_1/while/mul_3_grad/Mul/Enter:0
Mtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc:0Mtraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/Enter:0І
Qtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc:0Qtraining/Adam/gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/Enter:0
Mtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc:0Mtraining/Adam/gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/Enter:0
Mtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc:0Mtraining/Adam/gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/Enter:0z
;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/f_acc:0;training/Adam/gradients/lstm_1/while/mul_5_grad/Mul/Enter:0
Mtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc:0Mtraining/Adam/gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/Enter:0
Mtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc:0Mtraining/Adam/gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/Enter:0
Mtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc:0Mtraining/Adam/gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/Enter:0:
lstm_1/strided_slice_9:0lstm_1/while/BiasAdd_1/Enter:0
Mtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc:0Mtraining/Adam/gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/Enter:0~
=training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/f_acc:0=training/Adam/gradients/lstm_1/while/mul_5_grad/Mul_1/Enter:09
lstm_1/strided_slice_4:0lstm_1/while/MatMul_4/Enter:0~
=training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/f_acc:0=training/Adam/gradients/lstm_1/while/mul_3_grad/Mul_1/Enter:0;
lstm_1/strided_slice_10:0lstm_1/while/BiasAdd_2/Enter:05
lstm_1/strided_slice:0lstm_1/while/MatMul/Enter:0>
!lstm_1/while/maximum_iterations:0lstm_1/while/Less/Enter:0 
Ntraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc:0Ntraining/Adam/gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/Enter:0Ђ
Otraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc_1:0Otraining/Adam/gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/Enter_1:09
lstm_1/strided_slice_5:0lstm_1/while/MatMul_5/Enter:0
Ltraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc:0Ltraining/Adam/gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/Enter:0Њ
Straining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc:0Straining/Adam/gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/Enter:0в
gtraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0gtraining/Adam/gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0@
lstm_1/TensorArray_1:0&lstm_1/while/TensorArrayReadV3/Enter:0
Ctraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_acc:0Ctraining/Adam/gradients/lstm_1/while/MatMul_3_grad/MatMul_1/Enter:0;
lstm_1/strided_slice_11:0lstm_1/while/BiasAdd_3/Enter:0Ђ
Otraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc_1:0Otraining/Adam/gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/Enter_1:0z
;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/f_acc:0;training/Adam/gradients/lstm_1/while/mul_2_grad/Mul/Enter:0Т
_training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_acc:0_training/Adam/gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/Enter:0o
Clstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0(lstm_1/while/TensorArrayReadV3/Enter_1:0Rlstm_1/while/Enter:0Rlstm_1/while/Enter_1:0Rlstm_1/while/Enter_2:0Rlstm_1/while/Enter_3:0Rlstm_1/while/Enter_4:0R#training/Adam/gradients/f_count_1:0Z!lstm_1/while/maximum_iterations:0"А$
	variablesЂ$$
X
lstm_1/kernel:0lstm_1/kernel/Assignlstm_1/kernel/read:02lstm_1/random_uniform:08

lstm_1/recurrent_kernel:0lstm_1/recurrent_kernel/Assignlstm_1/recurrent_kernel/read:02'lstm_1/recurrent_kernel/initial_value:08
J
lstm_1/bias:0lstm_1/bias/Assignlstm_1/bias/read:02lstm_1/concat:08
\
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02dense_1/random_uniform:08
M
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02dense_1/Const:08
\
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02dense_2/random_uniform:08
M
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02dense_2/Const:08
\
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:02dense_3/random_uniform:08
M
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:02dense_3/Const:08
f
Adam/iterations:0Adam/iterations/AssignAdam/iterations/read:02Adam/iterations/initial_value:08
F
	Adam/lr:0Adam/lr/AssignAdam/lr/read:02Adam/lr/initial_value:08
V
Adam/beta_1:0Adam/beta_1/AssignAdam/beta_1/read:02Adam/beta_1/initial_value:08
V
Adam/beta_2:0Adam/beta_2/AssignAdam/beta_2/read:02Adam/beta_2/initial_value:08
R
Adam/decay:0Adam/decay/AssignAdam/decay/read:02Adam/decay/initial_value:08
q
training/Adam/Variable:0training/Adam/Variable/Assigntraining/Adam/Variable/read:02training/Adam/zeros:08
y
training/Adam/Variable_1:0training/Adam/Variable_1/Assigntraining/Adam/Variable_1/read:02training/Adam/zeros_1:08
y
training/Adam/Variable_2:0training/Adam/Variable_2/Assigntraining/Adam/Variable_2/read:02training/Adam/zeros_2:08
y
training/Adam/Variable_3:0training/Adam/Variable_3/Assigntraining/Adam/Variable_3/read:02training/Adam/zeros_3:08
y
training/Adam/Variable_4:0training/Adam/Variable_4/Assigntraining/Adam/Variable_4/read:02training/Adam/zeros_4:08
y
training/Adam/Variable_5:0training/Adam/Variable_5/Assigntraining/Adam/Variable_5/read:02training/Adam/zeros_5:08
y
training/Adam/Variable_6:0training/Adam/Variable_6/Assigntraining/Adam/Variable_6/read:02training/Adam/zeros_6:08
y
training/Adam/Variable_7:0training/Adam/Variable_7/Assigntraining/Adam/Variable_7/read:02training/Adam/zeros_7:08
y
training/Adam/Variable_8:0training/Adam/Variable_8/Assigntraining/Adam/Variable_8/read:02training/Adam/zeros_8:08
y
training/Adam/Variable_9:0training/Adam/Variable_9/Assigntraining/Adam/Variable_9/read:02training/Adam/zeros_9:08
}
training/Adam/Variable_10:0 training/Adam/Variable_10/Assign training/Adam/Variable_10/read:02training/Adam/zeros_10:08
}
training/Adam/Variable_11:0 training/Adam/Variable_11/Assign training/Adam/Variable_11/read:02training/Adam/zeros_11:08
}
training/Adam/Variable_12:0 training/Adam/Variable_12/Assign training/Adam/Variable_12/read:02training/Adam/zeros_12:08
}
training/Adam/Variable_13:0 training/Adam/Variable_13/Assign training/Adam/Variable_13/read:02training/Adam/zeros_13:08
}
training/Adam/Variable_14:0 training/Adam/Variable_14/Assign training/Adam/Variable_14/read:02training/Adam/zeros_14:08
}
training/Adam/Variable_15:0 training/Adam/Variable_15/Assign training/Adam/Variable_15/read:02training/Adam/zeros_15:08
}
training/Adam/Variable_16:0 training/Adam/Variable_16/Assign training/Adam/Variable_16/read:02training/Adam/zeros_16:08
}
training/Adam/Variable_17:0 training/Adam/Variable_17/Assign training/Adam/Variable_17/read:02training/Adam/zeros_17:08
}
training/Adam/Variable_18:0 training/Adam/Variable_18/Assign training/Adam/Variable_18/read:02training/Adam/zeros_18:08
}
training/Adam/Variable_19:0 training/Adam/Variable_19/Assign training/Adam/Variable_19/read:02training/Adam/zeros_19:08
}
training/Adam/Variable_20:0 training/Adam/Variable_20/Assign training/Adam/Variable_20/read:02training/Adam/zeros_20:08
}
training/Adam/Variable_21:0 training/Adam/Variable_21/Assign training/Adam/Variable_21/read:02training/Adam/zeros_21:08
}
training/Adam/Variable_22:0 training/Adam/Variable_22/Assign training/Adam/Variable_22/read:02training/Adam/zeros_22:08
}
training/Adam/Variable_23:0 training/Adam/Variable_23/Assign training/Adam/Variable_23/read:02training/Adam/zeros_23:08
}
training/Adam/Variable_24:0 training/Adam/Variable_24/Assign training/Adam/Variable_24/read:02training/Adam/zeros_24:08
}
training/Adam/Variable_25:0 training/Adam/Variable_25/Assign training/Adam/Variable_25/read:02training/Adam/zeros_25:08
}
training/Adam/Variable_26:0 training/Adam/Variable_26/Assign training/Adam/Variable_26/read:02training/Adam/zeros_26:08        ЃK"	zўЉЭћжA*

losst?З<С       чЮј	ЌџЉЭћжA*


accют&?л y       и-	v?ЭћжA*

lossКJ?NNД       ё(	w?ЭћжA*


accSC?ЕИмо       и-	 ЭћжA*

lossj.?ЕsЗ       ё(	ЭћжA*


acc#I?­ёX       и-	ШГДЭћжA*

lossЏ#"?ЃJБ_       ё(	ъДДЭћжA*


accyvJ?:ЋЗ       и-	E/cЭћжA*

loss §?8DА       ё(	Z0cЭћжA*


acc~N? йЏ       и-	ВЂxЭћжA*

lossљ?ЭеЁ       ё(	aЄxЭћжA*


accR?pѓj       и-	~уѓЭћжA*

lossСQ?XЛ;       ё(	ЄфѓЭћжA*


accR?Фи-Є       и-	hwЯ ЭћжA*

lossСy?#ЈЂ       ё(	ћxЯ ЭћжA*


accНU?JЕіp       и-	_GYЄЭћжA*

lossћ?|ЙT       ё(	ЊHYЄЭћжA*


accй4V?REЏi       и-	ІсЇЭћжA	*

loss5лю>Wљ+0       ё(	бсЇЭћжA	*


acc/KZ?ляRЩ       и-	,+vЋЭћжA
*

lossCDв>YыЫ       ё(	,vЋЭћжA
*


accрЌ[?гЂЙВ       и-	5 ЏЭћжA*

lossРХ> ьhV       ё(	 ЏЭћжA*


acc	^?јE%Ђ       и-	РСВЭћжA*

loss&б>vЏЉ       ё(	еСВЭћжA*


accЋ\??       и-	Ю;LЖЭћжA*

loss?Ь>|№       ё(	ѓ<LЖЭћжA*


accul\?тWЇt       и-	<QФЙЭћжA*

lossъЛ>d9Л       ё(	\RФЙЭћжA*


accў~^?rgVЖ       и-	SВJНЭћжA*

loss&xЌ>rрр       ё(	ЃГJНЭћжA*


accQa?Л§O       и-	KщыРЭћжA*

lossUЕ>GiБ5       ё(	yъыРЭћжA*


acclя_? ГУ       и-	ФЭћжA*

lossjЌ>ш{:       ё(	6ФЭћжA*


accѕb?0ђX       и-	Ъ§ШЭћжA*

loss!&Њ>ќЎЎ       ё(	жўШЭћжA*


accЅKb?ЁgМF       и-	vЎЫЭћжA*

lossЂ>5$       ё(	`wЎЫЭћжA*


accІcc?1tнj       и-	SA2ЯЭћжA*

lossw>fм\В       ё(	ФB2ЯЭћжA*


accІcc?ЦWOj       и-	гЭћжA*

loss_I>8љМ`       ё(	MгЭћжA*


accдd?fя~Ќ       и-	б9КжЭћжA*

lossш1> >ЊV       ё(	Р;КжЭћжA*


accћaf?0       и-	КkVкЭћжA*

lossBB>\Ё       ё(	уlVкЭћжA*


acc!g?EЛh       и-	йъоЭћжA*

loss>ъuo       ё(	ЬьоЭћжA*


accsge?a$Щ        и-	*RжсЭћжA*

losszbw>z+       ё(	`SжсЭћжA*


acc3Оh?З7џ       и-	Z+iхЭћжA*

loss|Z>ыЂ)Ў       ё(	R,iхЭћжA*


acc?kg?жF       и-	^ЄњшЭћжA*

lossVхl>K       ё(	vЅњшЭћжA*


accoi?фЭд       и-	1?эЭћжA*

lossЫ>'џбЮ       ё(	l@эЭћжA*


accК h?}tVй       и-	m8ёЭћжA*

lossKѕc>Ъ^й       ё(	o8ёЭћжA*


acc))k?Е
       и-	фѕєЭћжA*

lossиГF>/HЧA       ё(	ЩхѕєЭћжA*


accМk?бФшH       и-	ЃфЉјЭћжA*

lossЗ%T>0wЭИ       ё(	-цЉјЭћжA*


acc*Al?6[m       и-	rМxќЭћжA *

lossВOZ>WMЭГ       ё(	НxќЭћжA *


accj?4Гќ       и-	ќЈ2 ЮћжA!*

loss~Z>ДЎѓ       ё(	G­2 ЮћжA!*


accCГj?JYwђ       и-	џYшЮћжA"*

lossVё]>ббЏ       ё(	>[шЮћжA"*


accЄj?Н        и-	ЦhЮћжA#*

loss3nE>лЫФ       ё(	ХЧhЮћжA#*


accМk?ЛV`с       и-	QNл
ЮћжA$*

lossDвB>ь1       ё(	Oл
ЮћжA$*


accкl?vDdO       и-	cbSЮћжA%*

lossЋB'>Ђ0Сt       ё(	kcSЮћжA%*


accЩn?ДЦ^[       и-	кЧЪЮћжA&*

lossQы;>ЖZb       ё(	ЮШЪЮћжA&*


accлЂm?ѕ){ђ       и-	Л ЮћжA'*

loss/>Qё       ё(	М ЮћжA'*


accq?wqр       и-	ЊЗЮћжA(*

loss;К1>йm       ё(	|ЙЮћжA(*


acc:p?с       и-	7ЮћжA)*

lossdЦ#>Мh       ё(	8ЮћжA)*


accТ0o?зs       и-	ЈуВЮћжA**

lossgл%>лжE{       ё(	бфВЮћжA**


accЕko?ЊїШЈ       и-	yаj#ЮћжA+*

loss@Б>СмrІ       ё(	бj#ЮћжA+*


accp?Яџj       и-	d'ЮћжA,*

lossЃ>*K       ё(	0e'ЮћжA,*


acc4q?^йс       и-	ЈWЎ*ЮћжA-*

lossР>hИь       ё(	ЩXЎ*ЮћжA-*


accпъp?Ю0       и-	ш1/.ЮћжA.*

lossш>VЉЬM       ё(	О2/.ЮћжA.*


acc=fp?шJа       и-	ы4Е1ЮћжA/*

lossVC
> З[Й       ё(	є5Е1ЮћжA/*


acc?~q?0~       и-	У)E5ЮћжA0*

lossѕ>008       ё(	В*E5ЮћжA0*


acc#мp?2ЉШЃ       и-	єNд8ЮћжA1*

loss{Ц>г       ё(	шOд8ЮћжA1*


accяпr?>wГ       и-	Х<ЮћжA2*

loss
ѓ=+vzC       ё(	ж<ЮћжA2*


acc@r?V4&       и-	-Г?ЮћжA3*

lossЕn>W<ј       ё(	Ў.Г?ЮћжA3*


accЌюr?4Oъ       и-	G 1CЮћжA4*

loss,_>ѕс       ё(	i1CЮћжA4*


acc:p?:ЁtH       и-	щeЋFЮћжA5*

lossнц><ьљ       ё(	4gЋFЮћжA5*


acc#мp?зА	       и-	ФyJЮћжA6*

lossѕ[>]@ћ!       ё(	ФzJЮћжA6*


acc\8s?{(Э       и-	'jMЮћжA7*

loss"ч>IGs       ё(	kMЮћжA7*


accLr?kка       и-	К1QЮћжA8*

loss`?>­       ё(	Ж2QЮћжA8*


accM[r?ЦuЦЉ       и-	TЮћжA9*

lossМz>РEЙ       ё(	ЇTЮћжA9*


acc$єq?( f       и-	VcXЮћжA:*

lossTXњ=Lт       ё(	VdXЮћжA:*


accvТr?ЌeР       и-	{І[ЮћжA;*

lossЄ&ы=.њ       ё(	І[ЮћжA;*


acc)s?вHY?       и-	ЕсC_ЮћжA<*

loss!9љ=ўЛT       ё(	тC_ЮћжA<*


acc§Єr?_Кј       и-	ћиbЮћжA=*

losssМ=67T       ё(	/ќиbЮћжA=*


acc`v?О№щ       и-	ЬafЮћжA>*

lossѓц=D        ё(	ЏЭafЮћжA>*


accжmt?Zdк       и-	:BЪiЮћжA?*

lossў>ЂJА       ё(	cCЪiЮћжA?*


accЦxr?њКЎo       и-	ЃЫQmЮћжA@*

loss'Nі=!Юџc       ё(	РЬQmЮћжA@*


accжmt?Т~Љл       и-	щ~qЮћжAA*

lossPЯ=э^E       ё(	AqЮћжAA*


accЂYu?;И2-       и-	7уСtЮћжAB*

lossZаЫ=/        ё(	/фСtЮћжAB*


accu?С§       и-	pxЮћжAC*

lossдlЮ=liХ       ё(	ћpxЮћжAC*


accQЃu?t       и-	Лб|ЮћжAD*

lossпк=eг       ё(	ев|ЮћжAD*


accиu?ћG       и-	ХУЮћжAE*

lossЁлМ=ЛRиЂ       ё(	ЇУЮћжAE*


accА6v?ѕђ8Ї       и-	#љ/ЮћжAF*

loss$ќЩ=ъ­Я       ё(	"ћ/ЮћжAF*


accl-u?К{­       и-	DФвЮћжAG*

lossX]э=%w       ё(	+ХвЮћжAG*


accЗt?<РА       и-	хd?ЮћжAH*

lossS >aoє       ё(	эe?ЮћжAH*


accЗt?Zђh       и-	ЦПШЮћжAI*

lossЕ= ОЉ       ё(	чРШЮћжAI*


accЃqv?С }        и-	H/sЮћжAJ*

loss4ђ=4&       ё(	#0sЮћжAJ*


accЗt?$iU       и-	 Ѓ/ЮћжAK*

loss*oЧ=ПхІ       ё(	Є/ЮћжAK*


acc7v?3л       и-	hЛЮћжAL*

losshЩ=G"y{       ё(	iЛЮћжAL*


accЌv?јё3       и-	3к9ЮћжAM*

lossРbЁ=#=И       ё(	<л9ЮћжAM*


accФw?ЎпЃ       и-	zЋЌЮћжAN*

lossЛDТ=eQM       ё(	dЌЌЮћжAN*


accйv?ьeЮ^       и-	z ЃЮћжAO*

lossоBЊ=N№яЛ       ё(	` ЃЮћжAO*


accGx?&!x       и-	WІЮћжAP*

lossO9=щmГ       ё(	SІЮћжAP*


accР+x?;W       и-	ж6ЊЮћжAQ*

loss7С=ыTЎ       ё(	7ЊЮћжAQ*


accv?т]ф       и-	мdi­ЮћжAR*

loss?V­=ЛЌ "       ё(	Іei­ЮћжAR*


accР+x?NWј       и-	яpмАЮћжAS*

lossDІ=mSє       ё(	rмАЮћжAS*


accaw?е
Ћ       и-	№mFДЮћжAT*

loss4GЇ=ђЉZ       ё(	зnFДЮћжAT*


accЌv?Ћ)l       и-	єнМЗЮћжAU*

lossЁГЌ=ОћZф       ё(	ЮоМЗЮћжAU*


acc|"w?ж4       и-	С)ЛЮћжAV*

loss~Ё=мЅ0g       ё(	)ЛЮћжAV*


acccАx?FjM       и-	XОЮћжAW*

lossЃЂ=ЂЅЂ       ё(	!ОЮћжAW*


accПx?Ыь}       и-	QчСЮћжAX*

lossvtЇ=Z§Бy       ё(	TRчСЮћжAX*


accaw??w       и-	ТЄOХЮћжAY*

lossџnЈ=,њ       ё(	~ЅOХЮћжAY*


accіWx?О       и-	lФШЮћжAZ*

loss=TЄЬ}       ё(	~mФШЮћжAZ*


accмx?LOр       и-	 ~.ЬЮћжA[*

lossI=ј
       ё(	.ЬЮћжA[*


accџw?шAђ       и-	CЯЮћжA\*

lossRЮ=^>Ѕ       ё(	cDЯЮћжA\*


acc,x?%зГ       и-	9ыгЮћжA]*

lossтbЂ=ТЇgЇ       ё(	6ьгЮћжA]*


accтw?Ђу)       и-	syжЮћжA^*

lossП0Є=u0U       ё(	^yжЮћжA^*


acc:Ix?фhJ       и-	NуйЮћжA_*

lossЯ=ѕRz       ё(	aOуйЮћжA_*


accy?Sд       и-	ЭWнЮћжA`*

lossfl=#їL       ё(	 WнЮћжA`*


accdШy?\м-       и-	6эрЮћжAa*

lossw=	&ч8       ё(	С7эрЮћжAa*


accіWx?C       и-	VюsфЮћжAb*

lossH=?ђ|И       ё(	ПяsфЮћжAb*


accыЊy?џJ       и-	ЦуѓчЮћжAc*

lossеyА=ш4H       ё(	СфѓчЮћжAc*


acc:Ix?kH