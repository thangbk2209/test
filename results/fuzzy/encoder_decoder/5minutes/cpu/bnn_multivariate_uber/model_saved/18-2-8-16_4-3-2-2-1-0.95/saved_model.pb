ль,
ј4у3
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
2	ђљ
Ь
	ApplyAdam
var"Tђ	
m"Tђ	
v"Tђ
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"Tђ" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"Tђ

value"T

output_ref"Tђ"	
Ttype"
validate_shapebool("
use_lockingbool(ў
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
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
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
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

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
:
Less
x"T
y"T
z
"
Ttype:
2	
$

LogicalAnd
x

y

z
љ
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

2	љ
Ї
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ
;
Minimum
x"T
y"T
z"T"
Ttype:

2	љ
=
Mul
x"T
y"T
z"T"
Ttype:
2	љ
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
Ї
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
2	ѕ
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
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
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
	elem_typetypeѕ
X
StackPushV2

handle	
elem"T
output"T"	
Ttype"
swap_memorybool( ѕ
S
StackV2
max_size

handle"
	elem_typetype"

stack_namestring ѕ
Ш
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
Є
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
ї
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
element_shapeshape:ѕ
`
TensorArrayGradV3

handle
flow_in
grad_handle
flow_out"
sourcestringѕ
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetypeѕ
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttypeѕ
9
TensorArraySizeV3

handle
flow_in
sizeѕ
я
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring ѕ
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttypeѕ
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
ref"dtypeђ"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ѕ
&
	ZerosLike
x"T
y"T"	
Ttype"tag*1.11.02v1.11.0-0-gc19e29306cњі'
v
PlaceholderPlaceholder*+
_output_shapes
:         * 
shape:         *
dtype0
x
Placeholder_1Placeholder* 
shape:         *
dtype0*+
_output_shapes
:         
p
Placeholder_2Placeholder*
shape:         *
dtype0*'
_output_shapes
:         
e
 encoder/DropoutWrapperInit/ConstConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
g
"encoder/DropoutWrapperInit/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *33s?
g
"encoder/DropoutWrapperInit/Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *33s?
a
encoder/concat/values_0Const*
valueB:*
dtype0*
_output_shapes
:
a
encoder/concat/values_1Const*
valueB:*
dtype0*
_output_shapes
:
U
encoder/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Џ
encoder/concatConcatV2encoder/concat/values_0encoder/concat/values_1encoder/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
_
encoder/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_
encoder/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
ћ
$encoder/random_uniform/RandomUniformRandomUniformencoder/concat*
dtype0*
_output_shapes

:*
seed2 *

seed *
T0
z
encoder/random_uniform/subSubencoder/random_uniform/maxencoder/random_uniform/min*
T0*
_output_shapes
: 
ї
encoder/random_uniform/mulMul$encoder/random_uniform/RandomUniformencoder/random_uniform/sub*
T0*
_output_shapes

:
~
encoder/random_uniformAddencoder/random_uniform/mulencoder/random_uniform/min*
_output_shapes

:*
T0
c
encoder/concat_1/values_0Const*
dtype0*
_output_shapes
:*
valueB:
c
encoder/concat_1/values_1Const*
valueB:*
dtype0*
_output_shapes
:
W
encoder/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Б
encoder/concat_1ConcatV2encoder/concat_1/values_0encoder/concat_1/values_1encoder/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
a
encoder/random_uniform_1/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
a
encoder/random_uniform_1/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
ў
&encoder/random_uniform_1/RandomUniformRandomUniformencoder/concat_1*
T0*
dtype0*
_output_shapes

:*
seed2 *

seed 
ђ
encoder/random_uniform_1/subSubencoder/random_uniform_1/maxencoder/random_uniform_1/min*
T0*
_output_shapes
: 
њ
encoder/random_uniform_1/mulMul&encoder/random_uniform_1/RandomUniformencoder/random_uniform_1/sub*
T0*
_output_shapes

:
ё
encoder/random_uniform_1Addencoder/random_uniform_1/mulencoder/random_uniform_1/min*
T0*
_output_shapes

:
c
encoder/concat_2/values_0Const*
valueB:*
dtype0*
_output_shapes
:
c
encoder/concat_2/values_1Const*
valueB:*
dtype0*
_output_shapes
:
W
encoder/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Б
encoder/concat_2ConcatV2encoder/concat_2/values_0encoder/concat_2/values_1encoder/concat_2/axis*
T0*
N*
_output_shapes
:*

Tidx0
a
encoder/random_uniform_2/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
a
encoder/random_uniform_2/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
ў
&encoder/random_uniform_2/RandomUniformRandomUniformencoder/concat_2*
dtype0*
_output_shapes

:*
seed2 *

seed *
T0
ђ
encoder/random_uniform_2/subSubencoder/random_uniform_2/maxencoder/random_uniform_2/min*
T0*
_output_shapes
: 
њ
encoder/random_uniform_2/mulMul&encoder/random_uniform_2/RandomUniformencoder/random_uniform_2/sub*
T0*
_output_shapes

:
ё
encoder/random_uniform_2Addencoder/random_uniform_2/mulencoder/random_uniform_2/min*
_output_shapes

:*
T0
g
"encoder/DropoutWrapperInit_1/ConstConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
i
$encoder/DropoutWrapperInit_1/Const_1Const*
_output_shapes
: *
valueB
 *33s?*
dtype0
i
$encoder/DropoutWrapperInit_1/Const_2Const*
valueB
 *33s?*
dtype0*
_output_shapes
: 
c
encoder/concat_3/values_0Const*
valueB:*
dtype0*
_output_shapes
:
c
encoder/concat_3/values_1Const*
valueB:*
dtype0*
_output_shapes
:
W
encoder/concat_3/axisConst*
_output_shapes
: *
value	B : *
dtype0
Б
encoder/concat_3ConcatV2encoder/concat_3/values_0encoder/concat_3/values_1encoder/concat_3/axis*
N*
_output_shapes
:*

Tidx0*
T0
a
encoder/random_uniform_3/minConst*
dtype0*
_output_shapes
: *
valueB
 *    
a
encoder/random_uniform_3/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
ў
&encoder/random_uniform_3/RandomUniformRandomUniformencoder/concat_3*
T0*
dtype0*
_output_shapes

:*
seed2 *

seed 
ђ
encoder/random_uniform_3/subSubencoder/random_uniform_3/maxencoder/random_uniform_3/min*
T0*
_output_shapes
: 
њ
encoder/random_uniform_3/mulMul&encoder/random_uniform_3/RandomUniformencoder/random_uniform_3/sub*
T0*
_output_shapes

:
ё
encoder/random_uniform_3Addencoder/random_uniform_3/mulencoder/random_uniform_3/min*
T0*
_output_shapes

:
c
encoder/concat_4/values_0Const*
valueB:*
dtype0*
_output_shapes
:
c
encoder/concat_4/values_1Const*
valueB:*
dtype0*
_output_shapes
:
W
encoder/concat_4/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Б
encoder/concat_4ConcatV2encoder/concat_4/values_0encoder/concat_4/values_1encoder/concat_4/axis*

Tidx0*
T0*
N*
_output_shapes
:
a
encoder/random_uniform_4/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
a
encoder/random_uniform_4/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
ў
&encoder/random_uniform_4/RandomUniformRandomUniformencoder/concat_4*
dtype0*
_output_shapes

:*
seed2 *

seed *
T0
ђ
encoder/random_uniform_4/subSubencoder/random_uniform_4/maxencoder/random_uniform_4/min*
_output_shapes
: *
T0
њ
encoder/random_uniform_4/mulMul&encoder/random_uniform_4/RandomUniformencoder/random_uniform_4/sub*
T0*
_output_shapes

:
ё
encoder/random_uniform_4Addencoder/random_uniform_4/mulencoder/random_uniform_4/min*
T0*
_output_shapes

:
c
encoder/concat_5/values_0Const*
dtype0*
_output_shapes
:*
valueB:
c
encoder/concat_5/values_1Const*
valueB:*
dtype0*
_output_shapes
:
W
encoder/concat_5/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Б
encoder/concat_5ConcatV2encoder/concat_5/values_0encoder/concat_5/values_1encoder/concat_5/axis*

Tidx0*
T0*
N*
_output_shapes
:
a
encoder/random_uniform_5/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
a
encoder/random_uniform_5/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
ў
&encoder/random_uniform_5/RandomUniformRandomUniformencoder/concat_5*

seed *
T0*
dtype0*
_output_shapes

:*
seed2 
ђ
encoder/random_uniform_5/subSubencoder/random_uniform_5/maxencoder/random_uniform_5/min*
_output_shapes
: *
T0
њ
encoder/random_uniform_5/mulMul&encoder/random_uniform_5/RandomUniformencoder/random_uniform_5/sub*
_output_shapes

:*
T0
ё
encoder/random_uniform_5Addencoder/random_uniform_5/mulencoder/random_uniform_5/min*
_output_shapes

:*
T0
c
encoder/concat_6/values_0Const*
valueB:*
dtype0*
_output_shapes
:
c
encoder/concat_6/values_1Const*
valueB:*
dtype0*
_output_shapes
:
W
encoder/concat_6/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Б
encoder/concat_6ConcatV2encoder/concat_6/values_0encoder/concat_6/values_1encoder/concat_6/axis*
T0*
N*
_output_shapes
:*

Tidx0
a
encoder/random_uniform_6/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
a
encoder/random_uniform_6/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
ў
&encoder/random_uniform_6/RandomUniformRandomUniformencoder/concat_6*

seed *
T0*
dtype0*
_output_shapes

:*
seed2 
ђ
encoder/random_uniform_6/subSubencoder/random_uniform_6/maxencoder/random_uniform_6/min*
T0*
_output_shapes
: 
њ
encoder/random_uniform_6/mulMul&encoder/random_uniform_6/RandomUniformencoder/random_uniform_6/sub*
T0*
_output_shapes

:
ё
encoder/random_uniform_6Addencoder/random_uniform_6/mulencoder/random_uniform_6/min*
_output_shapes

:*
T0
R
encoder/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Y
encoder/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
Y
encoder/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
є
encoder/rnn/rangeRangeencoder/rnn/range/startencoder/rnn/Rankencoder/rnn/range/delta*
_output_shapes
:*

Tidx0
l
encoder/rnn/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
Y
encoder/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
А
encoder/rnn/concatConcatV2encoder/rnn/concat/values_0encoder/rnn/rangeencoder/rnn/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
є
encoder/rnn/transpose	TransposePlaceholderencoder/rnn/concat*
T0*+
_output_shapes
:         *
Tperm0
f
encoder/rnn/ShapeShapeencoder/rnn/transpose*
T0*
out_type0*
_output_shapes
:
i
encoder/rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
k
!encoder/rnn/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
k
!encoder/rnn/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
х
encoder/rnn/strided_sliceStridedSliceencoder/rnn/Shapeencoder/rnn/strided_slice/stack!encoder/rnn/strided_slice/stack_1!encoder/rnn/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
ю
Zencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
ї
Vencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims
ExpandDimsencoder/rnn/strided_sliceZencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims/dim*
T0*
_output_shapes
:*

Tdim0
Џ
Qencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:
Ў
Wencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ю
Rencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concatConcatV2Vencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDimsQencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/ConstWencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
ю
Wencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╩
Qencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zerosFillRencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concatWencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zeros/Const*'
_output_shapes
:         *
T0*

index_type0
ъ
\encoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
љ
Xencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1
ExpandDimsencoder/rnn/strided_slice\encoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1/dim*
_output_shapes
:*

Tdim0*
T0
Ю
Sencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:
ъ
\encoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
љ
Xencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2
ExpandDimsencoder/rnn/strided_slice\encoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2/dim*
T0*
_output_shapes
:*

Tdim0
Ю
Sencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_2Const*
dtype0*
_output_shapes
:*
valueB:
Џ
Yencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ц
Tencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concat_1ConcatV2Xencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2Sencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_2Yencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concat_1/axis*
_output_shapes
:*

Tidx0*
T0*
N
ъ
Yencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
л
Sencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1FillTencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concat_1Yencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1/Const*'
_output_shapes
:         *
T0*

index_type0
ъ
\encoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3/dimConst*
value	B : *
dtype0*
_output_shapes
: 
љ
Xencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3
ExpandDimsencoder/rnn/strided_slice\encoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3/dim*
T0*
_output_shapes
:*

Tdim0
Ю
Sencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:
ъ
\encoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
љ
Xencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/ExpandDims
ExpandDimsencoder/rnn/strided_slice\encoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0
Ю
Sencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:
Џ
Yencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ц
Tencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/concatConcatV2Xencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/ExpandDimsSencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/ConstYencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
ъ
Yencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
л
Sencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/zerosFillTencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/concatYencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/zeros/Const*'
_output_shapes
:         *
T0*

index_type0
а
^encoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/ExpandDims_1/dimConst*
_output_shapes
: *
value	B : *
dtype0
ћ
Zencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/ExpandDims_1
ExpandDimsencoder/rnn/strided_slice^encoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/ExpandDims_1/dim*
T0*
_output_shapes
:*

Tdim0
Ъ
Uencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:
а
^encoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
ћ
Zencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/ExpandDims_2
ExpandDimsencoder/rnn/strided_slice^encoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/ExpandDims_2/dim*
_output_shapes
:*

Tdim0*
T0
Ъ
Uencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/Const_2Const*
dtype0*
_output_shapes
:*
valueB:
Ю
[encoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
г
Vencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/concat_1ConcatV2Zencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/ExpandDims_2Uencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/Const_2[encoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
а
[encoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
о
Uencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/zeros_1FillVencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/concat_1[encoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/zeros_1/Const*
T0*

index_type0*'
_output_shapes
:         
а
^encoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/ExpandDims_3/dimConst*
_output_shapes
: *
value	B : *
dtype0
ћ
Zencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/ExpandDims_3
ExpandDimsencoder/rnn/strided_slice^encoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/ExpandDims_3/dim*
T0*
_output_shapes
:*

Tdim0
Ъ
Uencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:
h
encoder/rnn/Shape_1Shapeencoder/rnn/transpose*
T0*
out_type0*
_output_shapes
:
k
!encoder/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
m
#encoder/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
m
#encoder/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┐
encoder/rnn/strided_slice_1StridedSliceencoder/rnn/Shape_1!encoder/rnn/strided_slice_1/stack#encoder/rnn/strided_slice_1/stack_1#encoder/rnn/strided_slice_1/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
h
encoder/rnn/Shape_2Shapeencoder/rnn/transpose*
T0*
out_type0*
_output_shapes
:
k
!encoder/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
m
#encoder/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
m
#encoder/rnn/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
┐
encoder/rnn/strided_slice_2StridedSliceencoder/rnn/Shape_2!encoder/rnn/strided_slice_2/stack#encoder/rnn/strided_slice_2/stack_1#encoder/rnn/strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
\
encoder/rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
ј
encoder/rnn/ExpandDims
ExpandDimsencoder/rnn/strided_slice_2encoder/rnn/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
[
encoder/rnn/ConstConst*
valueB:*
dtype0*
_output_shapes
:
[
encoder/rnn/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
а
encoder/rnn/concat_1ConcatV2encoder/rnn/ExpandDimsencoder/rnn/Constencoder/rnn/concat_1/axis*
_output_shapes
:*

Tidx0*
T0*
N
\
encoder/rnn/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
ї
encoder/rnn/zerosFillencoder/rnn/concat_1encoder/rnn/zeros/Const*
T0*

index_type0*'
_output_shapes
:         
R
encoder/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
Џ
encoder/rnn/TensorArrayTensorArrayV3encoder/rnn/strided_slice_1*
identical_element_shapes(*7
tensor_array_name" encoder/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *$
element_shape:         *
clear_after_read(*
dynamic_size( 
ю
encoder/rnn/TensorArray_1TensorArrayV3encoder/rnn/strided_slice_1*6
tensor_array_name!encoder/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *$
element_shape:         *
dynamic_size( *
clear_after_read(*
identical_element_shapes(
y
$encoder/rnn/TensorArrayUnstack/ShapeShapeencoder/rnn/transpose*
T0*
out_type0*
_output_shapes
:
|
2encoder/rnn/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
~
4encoder/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
~
4encoder/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ћ
,encoder/rnn/TensorArrayUnstack/strided_sliceStridedSlice$encoder/rnn/TensorArrayUnstack/Shape2encoder/rnn/TensorArrayUnstack/strided_slice/stack4encoder/rnn/TensorArrayUnstack/strided_slice/stack_14encoder/rnn/TensorArrayUnstack/strided_slice/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
l
*encoder/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
l
*encoder/rnn/TensorArrayUnstack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
С
$encoder/rnn/TensorArrayUnstack/rangeRange*encoder/rnn/TensorArrayUnstack/range/start,encoder/rnn/TensorArrayUnstack/strided_slice*encoder/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:         *

Tidx0
ъ
Fencoder/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3encoder/rnn/TensorArray_1$encoder/rnn/TensorArrayUnstack/rangeencoder/rnn/transposeencoder/rnn/TensorArray_1:1*
T0*(
_class
loc:@encoder/rnn/transpose*
_output_shapes
: 
W
encoder/rnn/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
s
encoder/rnn/MaximumMaximumencoder/rnn/Maximum/xencoder/rnn/strided_slice_1*
T0*
_output_shapes
: 
q
encoder/rnn/MinimumMinimumencoder/rnn/strided_slice_1encoder/rnn/Maximum*
_output_shapes
: *
T0
e
#encoder/rnn/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
┼
encoder/rnn/while/EnterEnter#encoder/rnn/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: */

frame_name!encoder/rnn/while/while_context
┤
encoder/rnn/while/Enter_1Enterencoder/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: */

frame_name!encoder/rnn/while/while_context
й
encoder/rnn/while/Enter_2Enterencoder/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: */

frame_name!encoder/rnn/while/while_context
є
encoder/rnn/while/Enter_3EnterQencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zeros*
parallel_iterations *'
_output_shapes
:         */

frame_name!encoder/rnn/while/while_context*
T0*
is_constant( 
ѕ
encoder/rnn/while/Enter_4EnterSencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1*
parallel_iterations *'
_output_shapes
:         */

frame_name!encoder/rnn/while/while_context*
T0*
is_constant( 
ѕ
encoder/rnn/while/Enter_5EnterSencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *'
_output_shapes
:         */

frame_name!encoder/rnn/while/while_context
і
encoder/rnn/while/Enter_6EnterUencoder/rnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/zeros_1*
is_constant( *
parallel_iterations *'
_output_shapes
:         */

frame_name!encoder/rnn/while/while_context*
T0
є
encoder/rnn/while/MergeMergeencoder/rnn/while/Enterencoder/rnn/while/NextIteration*
N*
_output_shapes
: : *
T0
ї
encoder/rnn/while/Merge_1Mergeencoder/rnn/while/Enter_1!encoder/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
ї
encoder/rnn/while/Merge_2Mergeencoder/rnn/while/Enter_2!encoder/rnn/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
Ю
encoder/rnn/while/Merge_3Mergeencoder/rnn/while/Enter_3!encoder/rnn/while/NextIteration_3*
T0*
N*)
_output_shapes
:         : 
Ю
encoder/rnn/while/Merge_4Mergeencoder/rnn/while/Enter_4!encoder/rnn/while/NextIteration_4*
N*)
_output_shapes
:         : *
T0
Ю
encoder/rnn/while/Merge_5Mergeencoder/rnn/while/Enter_5!encoder/rnn/while/NextIteration_5*
T0*
N*)
_output_shapes
:         : 
Ю
encoder/rnn/while/Merge_6Mergeencoder/rnn/while/Enter_6!encoder/rnn/while/NextIteration_6*
N*)
_output_shapes
:         : *
T0
v
encoder/rnn/while/LessLessencoder/rnn/while/Mergeencoder/rnn/while/Less/Enter*
T0*
_output_shapes
: 
┬
encoder/rnn/while/Less/EnterEnterencoder/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: */

frame_name!encoder/rnn/while/while_context
|
encoder/rnn/while/Less_1Lessencoder/rnn/while/Merge_1encoder/rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
╝
encoder/rnn/while/Less_1/EnterEnterencoder/rnn/Minimum*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: */

frame_name!encoder/rnn/while/while_context
t
encoder/rnn/while/LogicalAnd
LogicalAndencoder/rnn/while/Lessencoder/rnn/while/Less_1*
_output_shapes
: 
\
encoder/rnn/while/LoopCondLoopCondencoder/rnn/while/LogicalAnd*
_output_shapes
: 
д
encoder/rnn/while/SwitchSwitchencoder/rnn/while/Mergeencoder/rnn/while/LoopCond*
T0**
_class 
loc:@encoder/rnn/while/Merge*
_output_shapes
: : 
г
encoder/rnn/while/Switch_1Switchencoder/rnn/while/Merge_1encoder/rnn/while/LoopCond*
_output_shapes
: : *
T0*,
_class"
 loc:@encoder/rnn/while/Merge_1
г
encoder/rnn/while/Switch_2Switchencoder/rnn/while/Merge_2encoder/rnn/while/LoopCond*
T0*,
_class"
 loc:@encoder/rnn/while/Merge_2*
_output_shapes
: : 
╬
encoder/rnn/while/Switch_3Switchencoder/rnn/while/Merge_3encoder/rnn/while/LoopCond*
T0*,
_class"
 loc:@encoder/rnn/while/Merge_3*:
_output_shapes(
&:         :         
╬
encoder/rnn/while/Switch_4Switchencoder/rnn/while/Merge_4encoder/rnn/while/LoopCond*,
_class"
 loc:@encoder/rnn/while/Merge_4*:
_output_shapes(
&:         :         *
T0
╬
encoder/rnn/while/Switch_5Switchencoder/rnn/while/Merge_5encoder/rnn/while/LoopCond*:
_output_shapes(
&:         :         *
T0*,
_class"
 loc:@encoder/rnn/while/Merge_5
╬
encoder/rnn/while/Switch_6Switchencoder/rnn/while/Merge_6encoder/rnn/while/LoopCond*
T0*,
_class"
 loc:@encoder/rnn/while/Merge_6*:
_output_shapes(
&:         :         
c
encoder/rnn/while/IdentityIdentityencoder/rnn/while/Switch:1*
T0*
_output_shapes
: 
g
encoder/rnn/while/Identity_1Identityencoder/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
g
encoder/rnn/while/Identity_2Identityencoder/rnn/while/Switch_2:1*
T0*
_output_shapes
: 
x
encoder/rnn/while/Identity_3Identityencoder/rnn/while/Switch_3:1*'
_output_shapes
:         *
T0
x
encoder/rnn/while/Identity_4Identityencoder/rnn/while/Switch_4:1*'
_output_shapes
:         *
T0
x
encoder/rnn/while/Identity_5Identityencoder/rnn/while/Switch_5:1*
T0*'
_output_shapes
:         
x
encoder/rnn/while/Identity_6Identityencoder/rnn/while/Switch_6:1*
T0*'
_output_shapes
:         
v
encoder/rnn/while/add/yConst^encoder/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
r
encoder/rnn/while/addAddencoder/rnn/while/Identityencoder/rnn/while/add/y*
_output_shapes
: *
T0
С
#encoder/rnn/while/TensorArrayReadV3TensorArrayReadV3)encoder/rnn/while/TensorArrayReadV3/Enterencoder/rnn/while/Identity_1+encoder/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:         
Л
)encoder/rnn/while/TensorArrayReadV3/EnterEnterencoder/rnn/TensorArray_1*
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
Ч
+encoder/rnn/while/TensorArrayReadV3/Enter_1EnterFencoder/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: */

frame_name!encoder/rnn/while/while_context
в
Sencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/shapeConst*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
valueB"   @   *
dtype0*
_output_shapes
:
П
Qencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/minConst*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
valueB
 *іЙ*
dtype0*
_output_shapes
: 
П
Qencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
valueB
 *і>*
dtype0
О
[encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformSencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/shape*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
seed2 *
dtype0*
_output_shapes

:@*

seed 
Т
Qencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/subSubQencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/maxQencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/min*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
_output_shapes
: 
Э
Qencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/mulMul[encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/RandomUniformQencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/sub*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
_output_shapes

:@
Ж
Mencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniformAddQencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/mulQencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/min*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
_output_shapes

:@
ь
2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
VariableV2*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name 
▀
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AssignAssign2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelMencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
validate_shape(*
_output_shapes

:@
а
7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/readIdentity2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
T0*
_output_shapes

:@
н
Bencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:@*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
valueB@*    
р
0encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias
VariableV2*
shared_name *C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
╩
7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AssignAssign0encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasBencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes
:@
ў
5encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/readIdentity0encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
T0*
_output_shapes
:@
а
Aencoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat/axisConst^encoder/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Ћ
<encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concatConcatV2#encoder/rnn/while/TensorArrayReadV3encoder/rnn/while/Identity_4Aencoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:         
а
<encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMulMatMul<encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concatBencoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter*'
_output_shapes
:         @*
transpose_a( *
transpose_b( *
T0
ї
Bencoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/EnterEnter7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:@*/

frame_name!encoder/rnn/while/while_context
ћ
=encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAddBiasAdd<encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMulCencoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*'
_output_shapes
:         @
Є
Cencoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/EnterEnter5encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read*
parallel_iterations *
_output_shapes
:@*/

frame_name!encoder/rnn/while/while_context*
T0*
is_constant(
џ
;encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/ConstConst^encoder/rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
ц
Eencoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split/split_dimConst^encoder/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
к
;encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/splitSplitEencoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split/split_dim=encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd*
T0*`
_output_shapesN
L:         :         :         :         *
	num_split
Ю
;encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add/yConst^encoder/rnn/while/Identity*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
Ь
9encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/addAdd=encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:2;encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add/y*
T0*'
_output_shapes
:         
х
=encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/SigmoidSigmoid9encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add*
T0*'
_output_shapes
:         
¤
9encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mulMul=encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoidencoder/rnn/while/Identity_3*
T0*'
_output_shapes
:         
╣
?encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1Sigmoid;encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split*
T0*'
_output_shapes
:         
│
:encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/TanhTanh=encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:1*
T0*'
_output_shapes
:         
ы
;encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1Mul?encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1:encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh*
T0*'
_output_shapes
:         
В
;encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1Add9encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul;encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1*
T0*'
_output_shapes
:         
╗
?encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2Sigmoid=encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:3*'
_output_shapes
:         *
T0
│
<encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1Tanh;encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1*'
_output_shapes
:         *
T0
з
;encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2Mul?encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2<encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
T0*'
_output_shapes
:         
Њ
1encoder/rnn/while/rnn/multi_rnn_cell/cell_0/add/xConst^encoder/rnn/while/Identity*
valueB
 *33s?*
dtype0*
_output_shapes
: 
╔
/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/addAdd1encoder/rnn/while/rnn/multi_rnn_cell/cell_0/add/x5encoder/rnn/while/rnn/multi_rnn_cell/cell_0/add/Enter*
T0*
_output_shapes

:
Я
5encoder/rnn/while/rnn/multi_rnn_cell/cell_0/add/EnterEnterencoder/random_uniform_1*
parallel_iterations *
_output_shapes

:*/

frame_name!encoder/rnn/while/while_context*
T0*
is_constant(
ћ
1encoder/rnn/while/rnn/multi_rnn_cell/cell_0/FloorFloor/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/add*
_output_shapes

:*
T0
Њ
1encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div/yConst^encoder/rnn/while/Identity*
_output_shapes
: *
valueB
 *33s?*
dtype0
▄
/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/divRealDiv;encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_21encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div/y*'
_output_shapes
:         *
T0
╠
/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mulMul/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div1encoder/rnn/while/rnn/multi_rnn_cell/cell_0/Floor*
T0*'
_output_shapes
:         
Ћ
3encoder/rnn/while/rnn/multi_rnn_cell/cell_0/add_1/xConst^encoder/rnn/while/Identity*
_output_shapes
: *
valueB
 *33s?*
dtype0
¤
1encoder/rnn/while/rnn/multi_rnn_cell/cell_0/add_1Add3encoder/rnn/while/rnn/multi_rnn_cell/cell_0/add_1/x7encoder/rnn/while/rnn/multi_rnn_cell/cell_0/add_1/Enter*
T0*
_output_shapes

:
Р
7encoder/rnn/while/rnn/multi_rnn_cell/cell_0/add_1/EnterEnterencoder/random_uniform_2*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:*/

frame_name!encoder/rnn/while/while_context
ў
3encoder/rnn/while/rnn/multi_rnn_cell/cell_0/Floor_1Floor1encoder/rnn/while/rnn/multi_rnn_cell/cell_0/add_1*
T0*
_output_shapes

:
Ћ
3encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1/yConst^encoder/rnn/while/Identity*
valueB
 *33s?*
dtype0*
_output_shapes
: 
Я
1encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1RealDiv;encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_23encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1/y*
T0*'
_output_shapes
:         
м
1encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1Mul1encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_13encoder/rnn/while/rnn/multi_rnn_cell/cell_0/Floor_1*
T0*'
_output_shapes
:         
Њ
1encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add/xConst^encoder/rnn/while/Identity*
valueB
 *33s?*
dtype0*
_output_shapes
: 
╔
/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/addAdd1encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add/x5encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add/Enter*
T0*
_output_shapes

:
Я
5encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add/EnterEnterencoder/random_uniform_3*
_output_shapes

:*/

frame_name!encoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
ћ
1encoder/rnn/while/rnn/multi_rnn_cell/cell_1/FloorFloor/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add*
_output_shapes

:*
T0
Њ
1encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div/yConst^encoder/rnn/while/Identity*
valueB
 *33s?*
dtype0*
_output_shapes
: 
м
/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/divRealDiv1encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_11encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div/y*
T0*'
_output_shapes
:         
╠
/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mulMul/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div1encoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor*
T0*'
_output_shapes
:         
в
Sencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
valueB"      *
dtype0
П
Qencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
valueB
 *ВЛЙ
П
Qencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/maxConst*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
valueB
 *ВЛ>*
dtype0*
_output_shapes
: 
О
[encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformSencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
seed2 
Т
Qencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/subSubQencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/maxQencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
Э
Qencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/mulMul[encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/RandomUniformQencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/sub*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
_output_shapes

:
Ж
Mencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniformAddQencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/mulQencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/min*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
_output_shapes

:
ь
2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
▀
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AssignAssign2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernelMencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform*
_output_shapes

:*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
validate_shape(
а
7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/readIdentity2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
T0*
_output_shapes

:
н
Bencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Initializer/zerosConst*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
valueB*    *
dtype0*
_output_shapes
:
р
0encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias
VariableV2*
shared_name *C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
	container *
shape:*
dtype0*
_output_shapes
:
╩
7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AssignAssign0encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasBencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
validate_shape(*
_output_shapes
:
ў
5encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/readIdentity0encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
_output_shapes
:*
T0
а
Aencoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat/axisConst^encoder/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
А
<encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concatConcatV2/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mulencoder/rnn/while/Identity_6Aencoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat/axis*
T0*
N*'
_output_shapes
:         *

Tidx0
а
<encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMulMatMul<encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concatBencoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter*'
_output_shapes
:         *
transpose_a( *
transpose_b( *
T0
ї
Bencoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/EnterEnter7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read*
parallel_iterations *
_output_shapes

:*/

frame_name!encoder/rnn/while/while_context*
T0*
is_constant(
ћ
=encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAddBiasAdd<encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMulCencoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*'
_output_shapes
:         
Є
Cencoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/EnterEnter5encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
џ
;encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/ConstConst^encoder/rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
ц
Eencoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split/split_dimConst^encoder/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
к
;encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/splitSplitEencoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split/split_dim=encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd*
T0*`
_output_shapesN
L:         :         :         :         *
	num_split
Ю
;encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add/yConst^encoder/rnn/while/Identity*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
Ь
9encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/addAdd=encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:2;encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add/y*'
_output_shapes
:         *
T0
х
=encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/SigmoidSigmoid9encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add*
T0*'
_output_shapes
:         
¤
9encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mulMul=encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoidencoder/rnn/while/Identity_5*
T0*'
_output_shapes
:         
╣
?encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1Sigmoid;encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split*
T0*'
_output_shapes
:         
│
:encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/TanhTanh=encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:1*
T0*'
_output_shapes
:         
ы
;encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1Mul?encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1:encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh*
T0*'
_output_shapes
:         
В
;encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1Add9encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul;encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1*
T0*'
_output_shapes
:         
╗
?encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2Sigmoid=encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:3*
T0*'
_output_shapes
:         
│
<encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1Tanh;encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1*'
_output_shapes
:         *
T0
з
;encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2Mul?encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2<encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*'
_output_shapes
:         *
T0
Ћ
3encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_1/xConst^encoder/rnn/while/Identity*
_output_shapes
: *
valueB
 *33s?*
dtype0
¤
1encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_1Add3encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_1/x7encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_1/Enter*
_output_shapes

:*
T0
Р
7encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_1/EnterEnterencoder/random_uniform_5*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:*/

frame_name!encoder/rnn/while/while_context
ў
3encoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor_1Floor1encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_1*
T0*
_output_shapes

:
Ћ
3encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1/yConst^encoder/rnn/while/Identity*
valueB
 *33s?*
dtype0*
_output_shapes
: 
Я
1encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1RealDiv;encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_23encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1/y*'
_output_shapes
:         *
T0
м
1encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1Mul1encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_13encoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor_1*'
_output_shapes
:         *
T0
Ћ
3encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_2/xConst^encoder/rnn/while/Identity*
valueB
 *33s?*
dtype0*
_output_shapes
: 
¤
1encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_2Add3encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_2/x7encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_2/Enter*
T0*
_output_shapes

:
Р
7encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_2/EnterEnterencoder/random_uniform_6*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:*/

frame_name!encoder/rnn/while/while_context
ў
3encoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor_2Floor1encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_2*
T0*
_output_shapes

:
Ћ
3encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2/yConst^encoder/rnn/while/Identity*
valueB
 *33s?*
dtype0*
_output_shapes
: 
Я
1encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2RealDiv;encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_23encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2/y*'
_output_shapes
:         *
T0
м
1encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2Mul1encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_23encoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor_2*'
_output_shapes
:         *
T0
я
5encoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3;encoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterencoder/rnn/while/Identity_11encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2encoder/rnn/while/Identity_2*D
_class:
86loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2*
_output_shapes
: *
T0
Д
;encoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterencoder/rnn/TensorArray*
parallel_iterations *
is_constant(*
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context*
T0*D
_class:
86loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2
x
encoder/rnn/while/add_1/yConst^encoder/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
x
encoder/rnn/while/add_1Addencoder/rnn/while/Identity_1encoder/rnn/while/add_1/y*
T0*
_output_shapes
: 
h
encoder/rnn/while/NextIterationNextIterationencoder/rnn/while/add*
_output_shapes
: *
T0
l
!encoder/rnn/while/NextIteration_1NextIterationencoder/rnn/while/add_1*
T0*
_output_shapes
: 
і
!encoder/rnn/while/NextIteration_2NextIteration5encoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
А
!encoder/rnn/while/NextIteration_3NextIteration;encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1*'
_output_shapes
:         *
T0
Ћ
!encoder/rnn/while/NextIteration_4NextIteration/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul*'
_output_shapes
:         *
T0
А
!encoder/rnn/while/NextIteration_5NextIteration;encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1*
T0*'
_output_shapes
:         
Ќ
!encoder/rnn/while/NextIteration_6NextIteration1encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1*
T0*'
_output_shapes
:         
Y
encoder/rnn/while/ExitExitencoder/rnn/while/Switch*
_output_shapes
: *
T0
]
encoder/rnn/while/Exit_1Exitencoder/rnn/while/Switch_1*
T0*
_output_shapes
: 
]
encoder/rnn/while/Exit_2Exitencoder/rnn/while/Switch_2*
_output_shapes
: *
T0
n
encoder/rnn/while/Exit_3Exitencoder/rnn/while/Switch_3*
T0*'
_output_shapes
:         
n
encoder/rnn/while/Exit_4Exitencoder/rnn/while/Switch_4*'
_output_shapes
:         *
T0
n
encoder/rnn/while/Exit_5Exitencoder/rnn/while/Switch_5*'
_output_shapes
:         *
T0
n
encoder/rnn/while/Exit_6Exitencoder/rnn/while/Switch_6*
T0*'
_output_shapes
:         
║
.encoder/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3encoder/rnn/TensorArrayencoder/rnn/while/Exit_2**
_class 
loc:@encoder/rnn/TensorArray*
_output_shapes
: 
ќ
(encoder/rnn/TensorArrayStack/range/startConst**
_class 
loc:@encoder/rnn/TensorArray*
value	B : *
dtype0*
_output_shapes
: 
ќ
(encoder/rnn/TensorArrayStack/range/deltaConst*
_output_shapes
: **
_class 
loc:@encoder/rnn/TensorArray*
value	B :*
dtype0
ї
"encoder/rnn/TensorArrayStack/rangeRange(encoder/rnn/TensorArrayStack/range/start.encoder/rnn/TensorArrayStack/TensorArraySizeV3(encoder/rnn/TensorArrayStack/range/delta**
_class 
loc:@encoder/rnn/TensorArray*#
_output_shapes
:         *

Tidx0
ф
0encoder/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3encoder/rnn/TensorArray"encoder/rnn/TensorArrayStack/rangeencoder/rnn/while/Exit_2*$
element_shape:         **
_class 
loc:@encoder/rnn/TensorArray*
dtype0*+
_output_shapes
:         
]
encoder/rnn/Const_1Const*
valueB:*
dtype0*
_output_shapes
:
T
encoder/rnn/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
[
encoder/rnn/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 
[
encoder/rnn/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ј
encoder/rnn/range_1Rangeencoder/rnn/range_1/startencoder/rnn/Rank_1encoder/rnn/range_1/delta*

Tidx0*
_output_shapes
:
n
encoder/rnn/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
[
encoder/rnn/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Е
encoder/rnn/concat_2ConcatV2encoder/rnn/concat_2/values_0encoder/rnn/range_1encoder/rnn/concat_2/axis*

Tidx0*
T0*
N*
_output_shapes
:
»
encoder/rnn/transpose_1	Transpose0encoder/rnn/TensorArrayStack/TensorArrayGatherV3encoder/rnn/concat_2*+
_output_shapes
:         *
Tperm0*
T0
r
encoder/outputs_encoderIdentityencoder/rnn/transpose_1*
T0*+
_output_shapes
:         
e
 decoder/DropoutWrapperInit/ConstConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
g
"decoder/DropoutWrapperInit/Const_1Const*
valueB
 *33s?*
dtype0*
_output_shapes
: 
g
"decoder/DropoutWrapperInit/Const_2Const*
valueB
 *33s?*
dtype0*
_output_shapes
: 
a
decoder/concat/values_0Const*
valueB:*
dtype0*
_output_shapes
:
a
decoder/concat/values_1Const*
valueB:*
dtype0*
_output_shapes
:
U
decoder/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Џ
decoder/concatConcatV2decoder/concat/values_0decoder/concat/values_1decoder/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
_
decoder/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_
decoder/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  ђ?*
dtype0
ћ
$decoder/random_uniform/RandomUniformRandomUniformdecoder/concat*
T0*
dtype0*
_output_shapes

:*
seed2 *

seed 
z
decoder/random_uniform/subSubdecoder/random_uniform/maxdecoder/random_uniform/min*
T0*
_output_shapes
: 
ї
decoder/random_uniform/mulMul$decoder/random_uniform/RandomUniformdecoder/random_uniform/sub*
T0*
_output_shapes

:
~
decoder/random_uniformAdddecoder/random_uniform/muldecoder/random_uniform/min*
T0*
_output_shapes

:
c
decoder/concat_1/values_0Const*
valueB:*
dtype0*
_output_shapes
:
c
decoder/concat_1/values_1Const*
valueB:*
dtype0*
_output_shapes
:
W
decoder/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Б
decoder/concat_1ConcatV2decoder/concat_1/values_0decoder/concat_1/values_1decoder/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
a
decoder/random_uniform_1/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
a
decoder/random_uniform_1/maxConst*
_output_shapes
: *
valueB
 *  ђ?*
dtype0
ў
&decoder/random_uniform_1/RandomUniformRandomUniformdecoder/concat_1*

seed *
T0*
dtype0*
_output_shapes

:*
seed2 
ђ
decoder/random_uniform_1/subSubdecoder/random_uniform_1/maxdecoder/random_uniform_1/min*
_output_shapes
: *
T0
њ
decoder/random_uniform_1/mulMul&decoder/random_uniform_1/RandomUniformdecoder/random_uniform_1/sub*
_output_shapes

:*
T0
ё
decoder/random_uniform_1Adddecoder/random_uniform_1/muldecoder/random_uniform_1/min*
T0*
_output_shapes

:
c
decoder/concat_2/values_0Const*
valueB:*
dtype0*
_output_shapes
:
c
decoder/concat_2/values_1Const*
valueB:*
dtype0*
_output_shapes
:
W
decoder/concat_2/axisConst*
_output_shapes
: *
value	B : *
dtype0
Б
decoder/concat_2ConcatV2decoder/concat_2/values_0decoder/concat_2/values_1decoder/concat_2/axis*
T0*
N*
_output_shapes
:*

Tidx0
a
decoder/random_uniform_2/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
a
decoder/random_uniform_2/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
ў
&decoder/random_uniform_2/RandomUniformRandomUniformdecoder/concat_2*

seed *
T0*
dtype0*
_output_shapes

:*
seed2 
ђ
decoder/random_uniform_2/subSubdecoder/random_uniform_2/maxdecoder/random_uniform_2/min*
T0*
_output_shapes
: 
њ
decoder/random_uniform_2/mulMul&decoder/random_uniform_2/RandomUniformdecoder/random_uniform_2/sub*
_output_shapes

:*
T0
ё
decoder/random_uniform_2Adddecoder/random_uniform_2/muldecoder/random_uniform_2/min*
T0*
_output_shapes

:
g
"decoder/DropoutWrapperInit_1/ConstConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
i
$decoder/DropoutWrapperInit_1/Const_1Const*
_output_shapes
: *
valueB
 *33s?*
dtype0
i
$decoder/DropoutWrapperInit_1/Const_2Const*
_output_shapes
: *
valueB
 *33s?*
dtype0
c
decoder/concat_3/values_0Const*
_output_shapes
:*
valueB:*
dtype0
c
decoder/concat_3/values_1Const*
valueB:*
dtype0*
_output_shapes
:
W
decoder/concat_3/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Б
decoder/concat_3ConcatV2decoder/concat_3/values_0decoder/concat_3/values_1decoder/concat_3/axis*
T0*
N*
_output_shapes
:*

Tidx0
a
decoder/random_uniform_3/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
a
decoder/random_uniform_3/maxConst*
_output_shapes
: *
valueB
 *  ђ?*
dtype0
ў
&decoder/random_uniform_3/RandomUniformRandomUniformdecoder/concat_3*
T0*
dtype0*
_output_shapes

:*
seed2 *

seed 
ђ
decoder/random_uniform_3/subSubdecoder/random_uniform_3/maxdecoder/random_uniform_3/min*
_output_shapes
: *
T0
њ
decoder/random_uniform_3/mulMul&decoder/random_uniform_3/RandomUniformdecoder/random_uniform_3/sub*
T0*
_output_shapes

:
ё
decoder/random_uniform_3Adddecoder/random_uniform_3/muldecoder/random_uniform_3/min*
T0*
_output_shapes

:
c
decoder/concat_4/values_0Const*
valueB:*
dtype0*
_output_shapes
:
c
decoder/concat_4/values_1Const*
_output_shapes
:*
valueB:*
dtype0
W
decoder/concat_4/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Б
decoder/concat_4ConcatV2decoder/concat_4/values_0decoder/concat_4/values_1decoder/concat_4/axis*
T0*
N*
_output_shapes
:*

Tidx0
a
decoder/random_uniform_4/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
a
decoder/random_uniform_4/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
ў
&decoder/random_uniform_4/RandomUniformRandomUniformdecoder/concat_4*
T0*
dtype0*
_output_shapes

:*
seed2 *

seed 
ђ
decoder/random_uniform_4/subSubdecoder/random_uniform_4/maxdecoder/random_uniform_4/min*
T0*
_output_shapes
: 
њ
decoder/random_uniform_4/mulMul&decoder/random_uniform_4/RandomUniformdecoder/random_uniform_4/sub*
T0*
_output_shapes

:
ё
decoder/random_uniform_4Adddecoder/random_uniform_4/muldecoder/random_uniform_4/min*
T0*
_output_shapes

:
c
decoder/concat_5/values_0Const*
valueB:*
dtype0*
_output_shapes
:
c
decoder/concat_5/values_1Const*
valueB:*
dtype0*
_output_shapes
:
W
decoder/concat_5/axisConst*
_output_shapes
: *
value	B : *
dtype0
Б
decoder/concat_5ConcatV2decoder/concat_5/values_0decoder/concat_5/values_1decoder/concat_5/axis*
_output_shapes
:*

Tidx0*
T0*
N
a
decoder/random_uniform_5/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
a
decoder/random_uniform_5/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
ў
&decoder/random_uniform_5/RandomUniformRandomUniformdecoder/concat_5*
T0*
dtype0*
_output_shapes

:*
seed2 *

seed 
ђ
decoder/random_uniform_5/subSubdecoder/random_uniform_5/maxdecoder/random_uniform_5/min*
_output_shapes
: *
T0
њ
decoder/random_uniform_5/mulMul&decoder/random_uniform_5/RandomUniformdecoder/random_uniform_5/sub*
T0*
_output_shapes

:
ё
decoder/random_uniform_5Adddecoder/random_uniform_5/muldecoder/random_uniform_5/min*
T0*
_output_shapes

:
c
decoder/concat_6/values_0Const*
dtype0*
_output_shapes
:*
valueB:
c
decoder/concat_6/values_1Const*
valueB:*
dtype0*
_output_shapes
:
W
decoder/concat_6/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Б
decoder/concat_6ConcatV2decoder/concat_6/values_0decoder/concat_6/values_1decoder/concat_6/axis*
N*
_output_shapes
:*

Tidx0*
T0
a
decoder/random_uniform_6/minConst*
_output_shapes
: *
valueB
 *    *
dtype0
a
decoder/random_uniform_6/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
ў
&decoder/random_uniform_6/RandomUniformRandomUniformdecoder/concat_6*
T0*
dtype0*
_output_shapes

:*
seed2 *

seed 
ђ
decoder/random_uniform_6/subSubdecoder/random_uniform_6/maxdecoder/random_uniform_6/min*
T0*
_output_shapes
: 
њ
decoder/random_uniform_6/mulMul&decoder/random_uniform_6/RandomUniformdecoder/random_uniform_6/sub*
T0*
_output_shapes

:
ё
decoder/random_uniform_6Adddecoder/random_uniform_6/muldecoder/random_uniform_6/min*
_output_shapes

:*
T0
R
decoder/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Y
decoder/rnn/range/startConst*
dtype0*
_output_shapes
: *
value	B :
Y
decoder/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
є
decoder/rnn/rangeRangedecoder/rnn/range/startdecoder/rnn/Rankdecoder/rnn/range/delta*

Tidx0*
_output_shapes
:
l
decoder/rnn/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
Y
decoder/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
А
decoder/rnn/concatConcatV2decoder/rnn/concat/values_0decoder/rnn/rangedecoder/rnn/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
ѕ
decoder/rnn/transpose	TransposePlaceholder_1decoder/rnn/concat*+
_output_shapes
:         *
Tperm0*
T0
f
decoder/rnn/ShapeShapedecoder/rnn/transpose*
T0*
out_type0*
_output_shapes
:
i
decoder/rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
k
!decoder/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
k
!decoder/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
х
decoder/rnn/strided_sliceStridedSlicedecoder/rnn/Shapedecoder/rnn/strided_slice/stack!decoder/rnn/strided_slice/stack_1!decoder/rnn/strided_slice/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
h
decoder/rnn/Shape_1Shapedecoder/rnn/transpose*
T0*
out_type0*
_output_shapes
:
k
!decoder/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
m
#decoder/rnn/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
m
#decoder/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┐
decoder/rnn/strided_slice_1StridedSlicedecoder/rnn/Shape_1!decoder/rnn/strided_slice_1/stack#decoder/rnn/strided_slice_1/stack_1#decoder/rnn/strided_slice_1/stack_2*
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask 
h
decoder/rnn/Shape_2Shapedecoder/rnn/transpose*
T0*
out_type0*
_output_shapes
:
k
!decoder/rnn/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:
m
#decoder/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
m
#decoder/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┐
decoder/rnn/strided_slice_2StridedSlicedecoder/rnn/Shape_2!decoder/rnn/strided_slice_2/stack#decoder/rnn/strided_slice_2/stack_1#decoder/rnn/strided_slice_2/stack_2*
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
\
decoder/rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
ј
decoder/rnn/ExpandDims
ExpandDimsdecoder/rnn/strided_slice_2decoder/rnn/ExpandDims/dim*
T0*
_output_shapes
:*

Tdim0
[
decoder/rnn/ConstConst*
valueB:*
dtype0*
_output_shapes
:
[
decoder/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
а
decoder/rnn/concat_1ConcatV2decoder/rnn/ExpandDimsdecoder/rnn/Constdecoder/rnn/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
\
decoder/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ї
decoder/rnn/zerosFilldecoder/rnn/concat_1decoder/rnn/zeros/Const*
T0*

index_type0*'
_output_shapes
:         
R
decoder/rnn/timeConst*
dtype0*
_output_shapes
: *
value	B : 
Џ
decoder/rnn/TensorArrayTensorArrayV3decoder/rnn/strided_slice_1*$
element_shape:         *
clear_after_read(*
dynamic_size( *
identical_element_shapes(*7
tensor_array_name" decoder/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: 
ю
decoder/rnn/TensorArray_1TensorArrayV3decoder/rnn/strided_slice_1*6
tensor_array_name!decoder/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *$
element_shape:         *
dynamic_size( *
clear_after_read(*
identical_element_shapes(
y
$decoder/rnn/TensorArrayUnstack/ShapeShapedecoder/rnn/transpose*
T0*
out_type0*
_output_shapes
:
|
2decoder/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
~
4decoder/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
~
4decoder/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ћ
,decoder/rnn/TensorArrayUnstack/strided_sliceStridedSlice$decoder/rnn/TensorArrayUnstack/Shape2decoder/rnn/TensorArrayUnstack/strided_slice/stack4decoder/rnn/TensorArrayUnstack/strided_slice/stack_14decoder/rnn/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
l
*decoder/rnn/TensorArrayUnstack/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
l
*decoder/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
С
$decoder/rnn/TensorArrayUnstack/rangeRange*decoder/rnn/TensorArrayUnstack/range/start,decoder/rnn/TensorArrayUnstack/strided_slice*decoder/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:         *

Tidx0
ъ
Fdecoder/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3decoder/rnn/TensorArray_1$decoder/rnn/TensorArrayUnstack/rangedecoder/rnn/transposedecoder/rnn/TensorArray_1:1*(
_class
loc:@decoder/rnn/transpose*
_output_shapes
: *
T0
W
decoder/rnn/Maximum/xConst*
_output_shapes
: *
value	B :*
dtype0
s
decoder/rnn/MaximumMaximumdecoder/rnn/Maximum/xdecoder/rnn/strided_slice_1*
T0*
_output_shapes
: 
q
decoder/rnn/MinimumMinimumdecoder/rnn/strided_slice_1decoder/rnn/Maximum*
T0*
_output_shapes
: 
e
#decoder/rnn/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
┼
decoder/rnn/while/EnterEnter#decoder/rnn/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: */

frame_name!decoder/rnn/while/while_context
┤
decoder/rnn/while/Enter_1Enterdecoder/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: */

frame_name!decoder/rnn/while/while_context
й
decoder/rnn/while/Enter_2Enterdecoder/rnn/TensorArray:1*
_output_shapes
: */

frame_name!decoder/rnn/while/while_context*
T0*
is_constant( *
parallel_iterations 
═
decoder/rnn/while/Enter_3Enterencoder/rnn/while/Exit_3*
T0*
is_constant( *
parallel_iterations *'
_output_shapes
:         */

frame_name!decoder/rnn/while/while_context
═
decoder/rnn/while/Enter_4Enterencoder/rnn/while/Exit_4*
is_constant( *
parallel_iterations *'
_output_shapes
:         */

frame_name!decoder/rnn/while/while_context*
T0
═
decoder/rnn/while/Enter_5Enterencoder/rnn/while/Exit_5*
T0*
is_constant( *
parallel_iterations *'
_output_shapes
:         */

frame_name!decoder/rnn/while/while_context
═
decoder/rnn/while/Enter_6Enterencoder/rnn/while/Exit_6*
T0*
is_constant( *
parallel_iterations *'
_output_shapes
:         */

frame_name!decoder/rnn/while/while_context
є
decoder/rnn/while/MergeMergedecoder/rnn/while/Enterdecoder/rnn/while/NextIteration*
T0*
N*
_output_shapes
: : 
ї
decoder/rnn/while/Merge_1Mergedecoder/rnn/while/Enter_1!decoder/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
ї
decoder/rnn/while/Merge_2Mergedecoder/rnn/while/Enter_2!decoder/rnn/while/NextIteration_2*
_output_shapes
: : *
T0*
N
Ю
decoder/rnn/while/Merge_3Mergedecoder/rnn/while/Enter_3!decoder/rnn/while/NextIteration_3*
T0*
N*)
_output_shapes
:         : 
Ю
decoder/rnn/while/Merge_4Mergedecoder/rnn/while/Enter_4!decoder/rnn/while/NextIteration_4*
T0*
N*)
_output_shapes
:         : 
Ю
decoder/rnn/while/Merge_5Mergedecoder/rnn/while/Enter_5!decoder/rnn/while/NextIteration_5*)
_output_shapes
:         : *
T0*
N
Ю
decoder/rnn/while/Merge_6Mergedecoder/rnn/while/Enter_6!decoder/rnn/while/NextIteration_6*
T0*
N*)
_output_shapes
:         : 
v
decoder/rnn/while/LessLessdecoder/rnn/while/Mergedecoder/rnn/while/Less/Enter*
T0*
_output_shapes
: 
┬
decoder/rnn/while/Less/EnterEnterdecoder/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: */

frame_name!decoder/rnn/while/while_context
|
decoder/rnn/while/Less_1Lessdecoder/rnn/while/Merge_1decoder/rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
╝
decoder/rnn/while/Less_1/EnterEnterdecoder/rnn/Minimum*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: */

frame_name!decoder/rnn/while/while_context
t
decoder/rnn/while/LogicalAnd
LogicalAnddecoder/rnn/while/Lessdecoder/rnn/while/Less_1*
_output_shapes
: 
\
decoder/rnn/while/LoopCondLoopConddecoder/rnn/while/LogicalAnd*
_output_shapes
: 
д
decoder/rnn/while/SwitchSwitchdecoder/rnn/while/Mergedecoder/rnn/while/LoopCond*
T0**
_class 
loc:@decoder/rnn/while/Merge*
_output_shapes
: : 
г
decoder/rnn/while/Switch_1Switchdecoder/rnn/while/Merge_1decoder/rnn/while/LoopCond*
T0*,
_class"
 loc:@decoder/rnn/while/Merge_1*
_output_shapes
: : 
г
decoder/rnn/while/Switch_2Switchdecoder/rnn/while/Merge_2decoder/rnn/while/LoopCond*
T0*,
_class"
 loc:@decoder/rnn/while/Merge_2*
_output_shapes
: : 
╬
decoder/rnn/while/Switch_3Switchdecoder/rnn/while/Merge_3decoder/rnn/while/LoopCond*
T0*,
_class"
 loc:@decoder/rnn/while/Merge_3*:
_output_shapes(
&:         :         
╬
decoder/rnn/while/Switch_4Switchdecoder/rnn/while/Merge_4decoder/rnn/while/LoopCond*
T0*,
_class"
 loc:@decoder/rnn/while/Merge_4*:
_output_shapes(
&:         :         
╬
decoder/rnn/while/Switch_5Switchdecoder/rnn/while/Merge_5decoder/rnn/while/LoopCond*
T0*,
_class"
 loc:@decoder/rnn/while/Merge_5*:
_output_shapes(
&:         :         
╬
decoder/rnn/while/Switch_6Switchdecoder/rnn/while/Merge_6decoder/rnn/while/LoopCond*,
_class"
 loc:@decoder/rnn/while/Merge_6*:
_output_shapes(
&:         :         *
T0
c
decoder/rnn/while/IdentityIdentitydecoder/rnn/while/Switch:1*
_output_shapes
: *
T0
g
decoder/rnn/while/Identity_1Identitydecoder/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
g
decoder/rnn/while/Identity_2Identitydecoder/rnn/while/Switch_2:1*
_output_shapes
: *
T0
x
decoder/rnn/while/Identity_3Identitydecoder/rnn/while/Switch_3:1*'
_output_shapes
:         *
T0
x
decoder/rnn/while/Identity_4Identitydecoder/rnn/while/Switch_4:1*'
_output_shapes
:         *
T0
x
decoder/rnn/while/Identity_5Identitydecoder/rnn/while/Switch_5:1*'
_output_shapes
:         *
T0
x
decoder/rnn/while/Identity_6Identitydecoder/rnn/while/Switch_6:1*'
_output_shapes
:         *
T0
v
decoder/rnn/while/add/yConst^decoder/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
r
decoder/rnn/while/addAdddecoder/rnn/while/Identitydecoder/rnn/while/add/y*
_output_shapes
: *
T0
С
#decoder/rnn/while/TensorArrayReadV3TensorArrayReadV3)decoder/rnn/while/TensorArrayReadV3/Enterdecoder/rnn/while/Identity_1+decoder/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:         
Л
)decoder/rnn/while/TensorArrayReadV3/EnterEnterdecoder/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
Ч
+decoder/rnn/while/TensorArrayReadV3/Enter_1EnterFdecoder/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: */

frame_name!decoder/rnn/while/while_context
в
Sdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/shapeConst*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
valueB"   @   *
dtype0*
_output_shapes
:
П
Qdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
valueB
 *іЙ
П
Qdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/maxConst*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
valueB
 *і>*
dtype0*
_output_shapes
: 
О
[decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformSdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes

:@*

seed *
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
Т
Qdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/subSubQdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/maxQdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/min*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
_output_shapes
: 
Э
Qdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/mulMul[decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/RandomUniformQdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/sub*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
_output_shapes

:@*
T0
Ж
Mdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniformAddQdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/mulQdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes

:@*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ь
2decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
VariableV2*
shared_name *E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
	container *
shape
:@*
dtype0*
_output_shapes

:@
▀
9decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AssignAssign2decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelMdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
а
7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/readIdentity2decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
T0*
_output_shapes

:@
н
Bdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/zerosConst*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
valueB@*    *
dtype0*
_output_shapes
:@
р
0decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias
VariableV2*
shared_name *C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
╩
7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AssignAssign0decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasBdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes
:@
ў
5decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/readIdentity0decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
_output_shapes
:@*
T0
а
Adecoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat/axisConst^decoder/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
Ћ
<decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concatConcatV2#decoder/rnn/while/TensorArrayReadV3decoder/rnn/while/Identity_4Adecoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat/axis*
T0*
N*'
_output_shapes
:         *

Tidx0
а
<decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMulMatMul<decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concatBdecoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b( 
ї
Bdecoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/EnterEnter7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:@*/

frame_name!decoder/rnn/while/while_context
ћ
=decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAddBiasAdd<decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMulCdecoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter*'
_output_shapes
:         @*
T0*
data_formatNHWC
Є
Cdecoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/EnterEnter5decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:@*/

frame_name!decoder/rnn/while/while_context
џ
;decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/ConstConst^decoder/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
ц
Edecoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split/split_dimConst^decoder/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
к
;decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/splitSplitEdecoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split/split_dim=decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd*`
_output_shapesN
L:         :         :         :         *
	num_split*
T0
Ю
;decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add/yConst^decoder/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
Ь
9decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/addAdd=decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:2;decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add/y*'
_output_shapes
:         *
T0
х
=decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/SigmoidSigmoid9decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add*
T0*'
_output_shapes
:         
¤
9decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mulMul=decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoiddecoder/rnn/while/Identity_3*
T0*'
_output_shapes
:         
╣
?decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1Sigmoid;decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split*'
_output_shapes
:         *
T0
│
:decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/TanhTanh=decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:1*
T0*'
_output_shapes
:         
ы
;decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1Mul?decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1:decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh*
T0*'
_output_shapes
:         
В
;decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1Add9decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul;decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1*
T0*'
_output_shapes
:         
╗
?decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2Sigmoid=decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:3*
T0*'
_output_shapes
:         
│
<decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1Tanh;decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1*'
_output_shapes
:         *
T0
з
;decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2Mul?decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2<decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
T0*'
_output_shapes
:         
Њ
1decoder/rnn/while/rnn/multi_rnn_cell/cell_0/add/xConst^decoder/rnn/while/Identity*
valueB
 *33s?*
dtype0*
_output_shapes
: 
╔
/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/addAdd1decoder/rnn/while/rnn/multi_rnn_cell/cell_0/add/x5decoder/rnn/while/rnn/multi_rnn_cell/cell_0/add/Enter*
T0*
_output_shapes

:
Я
5decoder/rnn/while/rnn/multi_rnn_cell/cell_0/add/EnterEnterdecoder/random_uniform_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:*/

frame_name!decoder/rnn/while/while_context
ћ
1decoder/rnn/while/rnn/multi_rnn_cell/cell_0/FloorFloor/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/add*
T0*
_output_shapes

:
Њ
1decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div/yConst^decoder/rnn/while/Identity*
valueB
 *33s?*
dtype0*
_output_shapes
: 
▄
/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/divRealDiv;decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_21decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div/y*'
_output_shapes
:         *
T0
╠
/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mulMul/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div1decoder/rnn/while/rnn/multi_rnn_cell/cell_0/Floor*
T0*'
_output_shapes
:         
Ћ
3decoder/rnn/while/rnn/multi_rnn_cell/cell_0/add_1/xConst^decoder/rnn/while/Identity*
valueB
 *33s?*
dtype0*
_output_shapes
: 
¤
1decoder/rnn/while/rnn/multi_rnn_cell/cell_0/add_1Add3decoder/rnn/while/rnn/multi_rnn_cell/cell_0/add_1/x7decoder/rnn/while/rnn/multi_rnn_cell/cell_0/add_1/Enter*
T0*
_output_shapes

:
Р
7decoder/rnn/while/rnn/multi_rnn_cell/cell_0/add_1/EnterEnterdecoder/random_uniform_2*
_output_shapes

:*/

frame_name!decoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
ў
3decoder/rnn/while/rnn/multi_rnn_cell/cell_0/Floor_1Floor1decoder/rnn/while/rnn/multi_rnn_cell/cell_0/add_1*
T0*
_output_shapes

:
Ћ
3decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1/yConst^decoder/rnn/while/Identity*
valueB
 *33s?*
dtype0*
_output_shapes
: 
Я
1decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1RealDiv;decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_23decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1/y*
T0*'
_output_shapes
:         
м
1decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1Mul1decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_13decoder/rnn/while/rnn/multi_rnn_cell/cell_0/Floor_1*
T0*'
_output_shapes
:         
Њ
1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add/xConst^decoder/rnn/while/Identity*
valueB
 *33s?*
dtype0*
_output_shapes
: 
╔
/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/addAdd1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add/x5decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add/Enter*
T0*
_output_shapes

:
Я
5decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add/EnterEnterdecoder/random_uniform_3*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:*/

frame_name!decoder/rnn/while/while_context
ћ
1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/FloorFloor/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add*
T0*
_output_shapes

:
Њ
1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div/yConst^decoder/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *33s?
м
/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/divRealDiv1decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_11decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div/y*'
_output_shapes
:         *
T0
╠
/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mulMul/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor*'
_output_shapes
:         *
T0
в
Sdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/shapeConst*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
valueB"      *
dtype0*
_output_shapes
:
П
Qdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/minConst*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
valueB
 *ВЛЙ*
dtype0*
_output_shapes
: 
П
Qdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/maxConst*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
valueB
 *ВЛ>*
dtype0*
_output_shapes
: 
О
[decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformSdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes

:*

seed *
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
Т
Qdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/subSubQdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/maxQdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/min*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
_output_shapes
: 
Э
Qdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/mulMul[decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/RandomUniformQdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/sub*
_output_shapes

:*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
Ж
Mdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniformAddQdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/mulQdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes

:*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ь
2decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
VariableV2*
dtype0*
_output_shapes

:*
shared_name *E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
	container *
shape
:
▀
9decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AssignAssign2decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernelMdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
validate_shape(*
_output_shapes

:
а
7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/readIdentity2decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
T0*
_output_shapes

:
н
Bdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Initializer/zerosConst*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
valueB*    *
dtype0*
_output_shapes
:
р
0decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
	container 
╩
7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AssignAssign0decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasBdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias
ў
5decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/readIdentity0decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
T0*
_output_shapes
:
а
Adecoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat/axisConst^decoder/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
А
<decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concatConcatV2/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/muldecoder/rnn/while/Identity_6Adecoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat/axis*
T0*
N*'
_output_shapes
:         *

Tidx0
а
<decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMulMatMul<decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concatBdecoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
ї
Bdecoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/EnterEnter7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:*/

frame_name!decoder/rnn/while/while_context
ћ
=decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAddBiasAdd<decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMulCdecoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*'
_output_shapes
:         
Є
Cdecoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/EnterEnter5decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
џ
;decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/ConstConst^decoder/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
ц
Edecoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split/split_dimConst^decoder/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
к
;decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/splitSplitEdecoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split/split_dim=decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd*
T0*`
_output_shapesN
L:         :         :         :         *
	num_split
Ю
;decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add/yConst^decoder/rnn/while/Identity*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
Ь
9decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/addAdd=decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:2;decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add/y*
T0*'
_output_shapes
:         
х
=decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/SigmoidSigmoid9decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add*'
_output_shapes
:         *
T0
¤
9decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mulMul=decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoiddecoder/rnn/while/Identity_5*
T0*'
_output_shapes
:         
╣
?decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1Sigmoid;decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split*
T0*'
_output_shapes
:         
│
:decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/TanhTanh=decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:1*'
_output_shapes
:         *
T0
ы
;decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1Mul?decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1:decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh*
T0*'
_output_shapes
:         
В
;decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1Add9decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul;decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1*
T0*'
_output_shapes
:         
╗
?decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2Sigmoid=decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:3*
T0*'
_output_shapes
:         
│
<decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1Tanh;decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1*
T0*'
_output_shapes
:         
з
;decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2Mul?decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2<decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*'
_output_shapes
:         *
T0
Ћ
3decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_1/xConst^decoder/rnn/while/Identity*
valueB
 *33s?*
dtype0*
_output_shapes
: 
¤
1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_1Add3decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_1/x7decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_1/Enter*
T0*
_output_shapes

:
Р
7decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_1/EnterEnterdecoder/random_uniform_5*
_output_shapes

:*/

frame_name!decoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
ў
3decoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor_1Floor1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_1*
T0*
_output_shapes

:
Ћ
3decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1/yConst^decoder/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *33s?
Я
1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1RealDiv;decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_23decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1/y*
T0*'
_output_shapes
:         
м
1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1Mul1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_13decoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor_1*
T0*'
_output_shapes
:         
Ћ
3decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_2/xConst^decoder/rnn/while/Identity*
_output_shapes
: *
valueB
 *33s?*
dtype0
¤
1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_2Add3decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_2/x7decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_2/Enter*
T0*
_output_shapes

:
Р
7decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_2/EnterEnterdecoder/random_uniform_6*
parallel_iterations *
_output_shapes

:*/

frame_name!decoder/rnn/while/while_context*
T0*
is_constant(
ў
3decoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor_2Floor1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_2*
_output_shapes

:*
T0
Ћ
3decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2/yConst^decoder/rnn/while/Identity*
valueB
 *33s?*
dtype0*
_output_shapes
: 
Я
1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2RealDiv;decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_23decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2/y*'
_output_shapes
:         *
T0
м
1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2Mul1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_23decoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor_2*'
_output_shapes
:         *
T0
я
5decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3;decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterdecoder/rnn/while/Identity_11decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2decoder/rnn/while/Identity_2*D
_class:
86loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2*
_output_shapes
: *
T0
Д
;decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterdecoder/rnn/TensorArray*
is_constant(*
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context*
T0*D
_class:
86loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2*
parallel_iterations 
x
decoder/rnn/while/add_1/yConst^decoder/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
x
decoder/rnn/while/add_1Adddecoder/rnn/while/Identity_1decoder/rnn/while/add_1/y*
T0*
_output_shapes
: 
h
decoder/rnn/while/NextIterationNextIterationdecoder/rnn/while/add*
T0*
_output_shapes
: 
l
!decoder/rnn/while/NextIteration_1NextIterationdecoder/rnn/while/add_1*
T0*
_output_shapes
: 
і
!decoder/rnn/while/NextIteration_2NextIteration5decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
А
!decoder/rnn/while/NextIteration_3NextIteration;decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1*
T0*'
_output_shapes
:         
Ћ
!decoder/rnn/while/NextIteration_4NextIteration/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul*
T0*'
_output_shapes
:         
А
!decoder/rnn/while/NextIteration_5NextIteration;decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1*'
_output_shapes
:         *
T0
Ќ
!decoder/rnn/while/NextIteration_6NextIteration1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1*'
_output_shapes
:         *
T0
Y
decoder/rnn/while/ExitExitdecoder/rnn/while/Switch*
T0*
_output_shapes
: 
]
decoder/rnn/while/Exit_1Exitdecoder/rnn/while/Switch_1*
_output_shapes
: *
T0
]
decoder/rnn/while/Exit_2Exitdecoder/rnn/while/Switch_2*
_output_shapes
: *
T0
n
decoder/rnn/while/Exit_3Exitdecoder/rnn/while/Switch_3*'
_output_shapes
:         *
T0
n
decoder/rnn/while/Exit_4Exitdecoder/rnn/while/Switch_4*'
_output_shapes
:         *
T0
n
decoder/rnn/while/Exit_5Exitdecoder/rnn/while/Switch_5*'
_output_shapes
:         *
T0
n
decoder/rnn/while/Exit_6Exitdecoder/rnn/while/Switch_6*
T0*'
_output_shapes
:         
║
.decoder/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3decoder/rnn/TensorArraydecoder/rnn/while/Exit_2**
_class 
loc:@decoder/rnn/TensorArray*
_output_shapes
: 
ќ
(decoder/rnn/TensorArrayStack/range/startConst*
dtype0*
_output_shapes
: **
_class 
loc:@decoder/rnn/TensorArray*
value	B : 
ќ
(decoder/rnn/TensorArrayStack/range/deltaConst**
_class 
loc:@decoder/rnn/TensorArray*
value	B :*
dtype0*
_output_shapes
: 
ї
"decoder/rnn/TensorArrayStack/rangeRange(decoder/rnn/TensorArrayStack/range/start.decoder/rnn/TensorArrayStack/TensorArraySizeV3(decoder/rnn/TensorArrayStack/range/delta*#
_output_shapes
:         *

Tidx0**
_class 
loc:@decoder/rnn/TensorArray
ф
0decoder/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3decoder/rnn/TensorArray"decoder/rnn/TensorArrayStack/rangedecoder/rnn/while/Exit_2*$
element_shape:         **
_class 
loc:@decoder/rnn/TensorArray*
dtype0*+
_output_shapes
:         
]
decoder/rnn/Const_1Const*
valueB:*
dtype0*
_output_shapes
:
T
decoder/rnn/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
[
decoder/rnn/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 
[
decoder/rnn/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ј
decoder/rnn/range_1Rangedecoder/rnn/range_1/startdecoder/rnn/Rank_1decoder/rnn/range_1/delta*
_output_shapes
:*

Tidx0
n
decoder/rnn/concat_2/values_0Const*
_output_shapes
:*
valueB"       *
dtype0
[
decoder/rnn/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Е
decoder/rnn/concat_2ConcatV2decoder/rnn/concat_2/values_0decoder/rnn/range_1decoder/rnn/concat_2/axis*
N*
_output_shapes
:*

Tidx0*
T0
»
decoder/rnn/transpose_1	Transpose0decoder/rnn/TensorArrayStack/TensorArrayGatherV3decoder/rnn/concat_2*
T0*+
_output_shapes
:         *
Tperm0
h
strided_slice/stackConst*
_output_shapes
:*!
valueB"            *
dtype0
j
strided_slice/stack_1Const*!
valueB"            *
dtype0*
_output_shapes
:
j
strided_slice/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
ю
strided_sliceStridedSlicedecoder/rnn/transpose_1strided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:         
Z
subSubPlaceholder_2strided_slice*
T0*'
_output_shapes
:         
G
SquareSquaresub*
T0*'
_output_shapes
:         
V
ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Y
MeanMeanSquareConst*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
S
gradients/f_countConst*
value	B : *
dtype0*
_output_shapes
: 
»
gradients/f_count_1Entergradients/f_count*
is_constant( *
parallel_iterations *
_output_shapes
: */

frame_name!decoder/rnn/while/while_context*
T0
r
gradients/MergeMergegradients/f_count_1gradients/NextIteration*
T0*
N*
_output_shapes
: : 
j
gradients/SwitchSwitchgradients/Mergedecoder/rnn/while/LoopCond*
T0*
_output_shapes
: : 
n
gradients/Add/yConst^decoder/rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
Z
gradients/AddAddgradients/Switch:1gradients/Add/y*
_output_shapes
: *
T0
▄.
gradients/NextIterationNextIterationgradients/Addc^gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2c^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/StackPushV2a^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/StackPushV2O^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/StackPushV2a^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPushV2m^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2o^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1k^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2_^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPushV2a^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPushV2_1m^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2o^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1[^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPushV2]^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPushV2m^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2o^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1[^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPushV2]^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPushV2k^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2m^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1Y^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPushV2[^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPushV2c^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/StackPushV2Q^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/StackPushV2S^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/StackPushV2a^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/StackPushV2O^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/StackPushV2Q^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/StackPushV2c^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/StackPushV2Q^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/StackPushV2c^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgs/StackPushV2a^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/StackPushV2O^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/StackPushV2a^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPushV2m^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2o^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1k^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2_^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPushV2a^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPushV2_1m^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2o^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1[^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPushV2]^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPushV2m^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2o^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1[^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPushV2]^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPushV2k^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2m^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1Y^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPushV2[^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPushV2c^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/StackPushV2Q^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/StackPushV2S^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/StackPushV2c^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgs/StackPushV2Q^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul/StackPushV2S^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul_1/StackPushV2a^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/StackPushV2O^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/StackPushV2Q^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/StackPushV2*
_output_shapes
: *
T0
N
gradients/f_count_2Exitgradients/Switch*
_output_shapes
: *
T0
S
gradients/b_countConst*
value	B :*
dtype0*
_output_shapes
: 
╗
gradients/b_count_1Entergradients/f_count_2*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *9

frame_name+)gradients/decoder/rnn/while/while_context
v
gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*
N*
_output_shapes
: : *
T0
x
gradients/GreaterEqualGreaterEqualgradients/Merge_1gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
┬
gradients/GreaterEqual/EnterEntergradients/b_count*
is_constant(*
parallel_iterations *
_output_shapes
: *9

frame_name+)gradients/decoder/rnn/while/while_context*
T0
O
gradients/b_count_2LoopCondgradients/GreaterEqual*
_output_shapes
: 
g
gradients/Switch_1Switchgradients/Merge_1gradients/b_count_2*
T0*
_output_shapes
: : 
i
gradients/SubSubgradients/Switch_1:1gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
║
gradients/NextIteration_1NextIterationgradients/Sub^^gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0*
_output_shapes
: 
P
gradients/b_count_3Exitgradients/Switch_1*
_output_shapes
: *
T0
U
gradients/f_count_3Const*
value	B : *
dtype0*
_output_shapes
: 
▒
gradients/f_count_4Entergradients/f_count_3*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: */

frame_name!encoder/rnn/while/while_context
v
gradients/Merge_2Mergegradients/f_count_4gradients/NextIteration_2*
N*
_output_shapes
: : *
T0
n
gradients/Switch_2Switchgradients/Merge_2encoder/rnn/while/LoopCond*
_output_shapes
: : *
T0
p
gradients/Add_1/yConst^encoder/rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
`
gradients/Add_1Addgradients/Switch_2:1gradients/Add_1/y*
T0*
_output_shapes
: 
Ѕ+
gradients/NextIteration_2NextIterationgradients/Add_1c^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/StackPushV2a^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/StackPushV2O^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/StackPushV2a^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPushV2m^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2o^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1k^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2_^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPushV2a^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPushV2_1m^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2o^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1[^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPushV2]^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPushV2m^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2o^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1[^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPushV2]^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPushV2k^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2m^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1Y^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPushV2[^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPushV2c^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/StackPushV2Q^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/StackPushV2S^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/StackPushV2a^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/StackPushV2O^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/StackPushV2Q^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/StackPushV2c^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/StackPushV2Q^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/StackPushV2a^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/StackPushV2O^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/StackPushV2a^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPushV2m^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2o^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1k^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2_^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPushV2a^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPushV2_1m^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2o^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1[^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPushV2]^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPushV2m^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2o^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1[^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPushV2]^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPushV2k^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2m^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1Y^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPushV2[^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPushV2c^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/StackPushV2Q^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/StackPushV2S^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/StackPushV2a^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/StackPushV2O^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/StackPushV2Q^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/StackPushV2*
_output_shapes
: *
T0
P
gradients/f_count_5Exitgradients/Switch_2*
_output_shapes
: *
T0
U
gradients/b_count_4Const*
value	B :*
dtype0*
_output_shapes
: 
╗
gradients/b_count_5Entergradients/f_count_5*
_output_shapes
: *9

frame_name+)gradients/encoder/rnn/while/while_context*
T0*
is_constant( *
parallel_iterations 
v
gradients/Merge_3Mergegradients/b_count_5gradients/NextIteration_3*
T0*
N*
_output_shapes
: : 
|
gradients/GreaterEqual_1GreaterEqualgradients/Merge_3gradients/GreaterEqual_1/Enter*
_output_shapes
: *
T0
к
gradients/GreaterEqual_1/EnterEntergradients/b_count_4*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *9

frame_name+)gradients/encoder/rnn/while/while_context
Q
gradients/b_count_6LoopCondgradients/GreaterEqual_1*
_output_shapes
: 
g
gradients/Switch_3Switchgradients/Merge_3gradients/b_count_6*
T0*
_output_shapes
: : 
m
gradients/Sub_1Subgradients/Switch_3:1gradients/GreaterEqual_1/Enter*
_output_shapes
: *
T0
║
gradients/NextIteration_3NextIterationgradients/Sub_1\^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/b_sync*
T0*
_output_shapes
: 
P
gradients/b_count_7Exitgradients/Switch_3*
_output_shapes
: *
T0
r
!gradients/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
љ
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
_
gradients/Mean_grad/ShapeShapeSquare*
T0*
out_type0*
_output_shapes
:
ю
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:         
a
gradients/Mean_grad/Shape_1ShapeSquare*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
ќ
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
џ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
_
gradients/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
ѓ
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
ђ
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
ї
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:         
~
gradients/Square_grad/ConstConst^gradients/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
t
gradients/Square_grad/MulMulsubgradients/Square_grad/Const*'
_output_shapes
:         *
T0
ї
gradients/Square_grad/Mul_1Mulgradients/Mean_grad/truedivgradients/Square_grad/Mul*
T0*'
_output_shapes
:         
e
gradients/sub_grad/ShapeShapePlaceholder_2*
T0*
out_type0*
_output_shapes
:
g
gradients/sub_grad/Shape_1Shapestrided_slice*
T0*
out_type0*
_output_shapes
:
┤
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:         :         
ц
gradients/sub_grad/SumSumgradients/Square_grad/Mul_1(gradients/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ќ
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
е
gradients/sub_grad/Sum_1Sumgradients/Square_grad/Mul_1*gradients/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_output_shapes
:*
T0
Џ
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
Tshape0*'
_output_shapes
:         *
T0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
┌
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*'
_output_shapes
:         
Я
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*'
_output_shapes
:         
y
"gradients/strided_slice_grad/ShapeShapedecoder/rnn/transpose_1*
T0*
out_type0*
_output_shapes
:
■
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad"gradients/strided_slice_grad/Shapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2-gradients/sub_grad/tuple/control_dependency_1*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*+
_output_shapes
:         
ѕ
8gradients/decoder/rnn/transpose_1_grad/InvertPermutationInvertPermutationdecoder/rnn/concat_2*
_output_shapes
:*
T0
ж
0gradients/decoder/rnn/transpose_1_grad/transpose	Transpose-gradients/strided_slice_grad/StridedSliceGrad8gradients/decoder/rnn/transpose_1_grad/InvertPermutation*
T0*+
_output_shapes
:         *
Tperm0
і
agradients/decoder/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3decoder/rnn/TensorArraydecoder/rnn/while/Exit_2*
_output_shapes

:: **
_class 
loc:@decoder/rnn/TensorArray*
source	gradients
┤
]gradients/decoder/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentitydecoder/rnn/while/Exit_2b^gradients/decoder/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0**
_class 
loc:@decoder/rnn/TensorArray
И
ggradients/decoder/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3agradients/decoder/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3"decoder/rnn/TensorArrayStack/range0gradients/decoder/rnn/transpose_1_grad/transpose]gradients/decoder/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
m
gradients/zeros_like	ZerosLikedecoder/rnn/while/Exit_3*'
_output_shapes
:         *
T0
o
gradients/zeros_like_1	ZerosLikedecoder/rnn/while/Exit_4*'
_output_shapes
:         *
T0
o
gradients/zeros_like_2	ZerosLikedecoder/rnn/while/Exit_5*'
_output_shapes
:         *
T0
o
gradients/zeros_like_3	ZerosLikedecoder/rnn/while/Exit_6*
T0*'
_output_shapes
:         
ф
.gradients/decoder/rnn/while/Exit_2_grad/b_exitEnterggradients/decoder/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *9

frame_name+)gradients/decoder/rnn/while/while_context
У
.gradients/decoder/rnn/while/Exit_3_grad/b_exitEntergradients/zeros_like*
T0*
is_constant( *
parallel_iterations *'
_output_shapes
:         *9

frame_name+)gradients/decoder/rnn/while/while_context
Ж
.gradients/decoder/rnn/while/Exit_4_grad/b_exitEntergradients/zeros_like_1*
is_constant( *
parallel_iterations *'
_output_shapes
:         *9

frame_name+)gradients/decoder/rnn/while/while_context*
T0
Ж
.gradients/decoder/rnn/while/Exit_5_grad/b_exitEntergradients/zeros_like_2*
parallel_iterations *'
_output_shapes
:         *9

frame_name+)gradients/decoder/rnn/while/while_context*
T0*
is_constant( 
Ж
.gradients/decoder/rnn/while/Exit_6_grad/b_exitEntergradients/zeros_like_3*
parallel_iterations *'
_output_shapes
:         *9

frame_name+)gradients/decoder/rnn/while/while_context*
T0*
is_constant( 
м
2gradients/decoder/rnn/while/Switch_2_grad/b_switchMerge.gradients/decoder/rnn/while/Exit_2_grad/b_exit9gradients/decoder/rnn/while/Switch_2_grad_1/NextIteration*
N*
_output_shapes
: : *
T0
с
2gradients/decoder/rnn/while/Switch_3_grad/b_switchMerge.gradients/decoder/rnn/while/Exit_3_grad/b_exit9gradients/decoder/rnn/while/Switch_3_grad_1/NextIteration*
N*)
_output_shapes
:         : *
T0
с
2gradients/decoder/rnn/while/Switch_4_grad/b_switchMerge.gradients/decoder/rnn/while/Exit_4_grad/b_exit9gradients/decoder/rnn/while/Switch_4_grad_1/NextIteration*
N*)
_output_shapes
:         : *
T0
с
2gradients/decoder/rnn/while/Switch_5_grad/b_switchMerge.gradients/decoder/rnn/while/Exit_5_grad/b_exit9gradients/decoder/rnn/while/Switch_5_grad_1/NextIteration*
N*)
_output_shapes
:         : *
T0
с
2gradients/decoder/rnn/while/Switch_6_grad/b_switchMerge.gradients/decoder/rnn/while/Exit_6_grad/b_exit9gradients/decoder/rnn/while/Switch_6_grad_1/NextIteration*
T0*
N*)
_output_shapes
:         : 
В
/gradients/decoder/rnn/while/Merge_2_grad/SwitchSwitch2gradients/decoder/rnn/while/Switch_2_grad/b_switchgradients/b_count_2*
_output_shapes
: : *
T0*E
_class;
97loc:@gradients/decoder/rnn/while/Switch_2_grad/b_switch
s
9gradients/decoder/rnn/while/Merge_2_grad/tuple/group_depsNoOp0^gradients/decoder/rnn/while/Merge_2_grad/Switch
б
Agradients/decoder/rnn/while/Merge_2_grad/tuple/control_dependencyIdentity/gradients/decoder/rnn/while/Merge_2_grad/Switch:^gradients/decoder/rnn/while/Merge_2_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/decoder/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
д
Cgradients/decoder/rnn/while/Merge_2_grad/tuple/control_dependency_1Identity1gradients/decoder/rnn/while/Merge_2_grad/Switch:1:^gradients/decoder/rnn/while/Merge_2_grad/tuple/group_deps*
_output_shapes
: *
T0*E
_class;
97loc:@gradients/decoder/rnn/while/Switch_2_grad/b_switch
ј
/gradients/decoder/rnn/while/Merge_3_grad/SwitchSwitch2gradients/decoder/rnn/while/Switch_3_grad/b_switchgradients/b_count_2*:
_output_shapes(
&:         :         *
T0*E
_class;
97loc:@gradients/decoder/rnn/while/Switch_3_grad/b_switch
s
9gradients/decoder/rnn/while/Merge_3_grad/tuple/group_depsNoOp0^gradients/decoder/rnn/while/Merge_3_grad/Switch
│
Agradients/decoder/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity/gradients/decoder/rnn/while/Merge_3_grad/Switch:^gradients/decoder/rnn/while/Merge_3_grad/tuple/group_deps*'
_output_shapes
:         *
T0*E
_class;
97loc:@gradients/decoder/rnn/while/Switch_3_grad/b_switch
и
Cgradients/decoder/rnn/while/Merge_3_grad/tuple/control_dependency_1Identity1gradients/decoder/rnn/while/Merge_3_grad/Switch:1:^gradients/decoder/rnn/while/Merge_3_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/decoder/rnn/while/Switch_3_grad/b_switch*'
_output_shapes
:         
ј
/gradients/decoder/rnn/while/Merge_4_grad/SwitchSwitch2gradients/decoder/rnn/while/Switch_4_grad/b_switchgradients/b_count_2*:
_output_shapes(
&:         :         *
T0*E
_class;
97loc:@gradients/decoder/rnn/while/Switch_4_grad/b_switch
s
9gradients/decoder/rnn/while/Merge_4_grad/tuple/group_depsNoOp0^gradients/decoder/rnn/while/Merge_4_grad/Switch
│
Agradients/decoder/rnn/while/Merge_4_grad/tuple/control_dependencyIdentity/gradients/decoder/rnn/while/Merge_4_grad/Switch:^gradients/decoder/rnn/while/Merge_4_grad/tuple/group_deps*E
_class;
97loc:@gradients/decoder/rnn/while/Switch_4_grad/b_switch*'
_output_shapes
:         *
T0
и
Cgradients/decoder/rnn/while/Merge_4_grad/tuple/control_dependency_1Identity1gradients/decoder/rnn/while/Merge_4_grad/Switch:1:^gradients/decoder/rnn/while/Merge_4_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/decoder/rnn/while/Switch_4_grad/b_switch*'
_output_shapes
:         
ј
/gradients/decoder/rnn/while/Merge_5_grad/SwitchSwitch2gradients/decoder/rnn/while/Switch_5_grad/b_switchgradients/b_count_2*:
_output_shapes(
&:         :         *
T0*E
_class;
97loc:@gradients/decoder/rnn/while/Switch_5_grad/b_switch
s
9gradients/decoder/rnn/while/Merge_5_grad/tuple/group_depsNoOp0^gradients/decoder/rnn/while/Merge_5_grad/Switch
│
Agradients/decoder/rnn/while/Merge_5_grad/tuple/control_dependencyIdentity/gradients/decoder/rnn/while/Merge_5_grad/Switch:^gradients/decoder/rnn/while/Merge_5_grad/tuple/group_deps*'
_output_shapes
:         *
T0*E
_class;
97loc:@gradients/decoder/rnn/while/Switch_5_grad/b_switch
и
Cgradients/decoder/rnn/while/Merge_5_grad/tuple/control_dependency_1Identity1gradients/decoder/rnn/while/Merge_5_grad/Switch:1:^gradients/decoder/rnn/while/Merge_5_grad/tuple/group_deps*E
_class;
97loc:@gradients/decoder/rnn/while/Switch_5_grad/b_switch*'
_output_shapes
:         *
T0
ј
/gradients/decoder/rnn/while/Merge_6_grad/SwitchSwitch2gradients/decoder/rnn/while/Switch_6_grad/b_switchgradients/b_count_2*
T0*E
_class;
97loc:@gradients/decoder/rnn/while/Switch_6_grad/b_switch*:
_output_shapes(
&:         :         
s
9gradients/decoder/rnn/while/Merge_6_grad/tuple/group_depsNoOp0^gradients/decoder/rnn/while/Merge_6_grad/Switch
│
Agradients/decoder/rnn/while/Merge_6_grad/tuple/control_dependencyIdentity/gradients/decoder/rnn/while/Merge_6_grad/Switch:^gradients/decoder/rnn/while/Merge_6_grad/tuple/group_deps*'
_output_shapes
:         *
T0*E
_class;
97loc:@gradients/decoder/rnn/while/Switch_6_grad/b_switch
и
Cgradients/decoder/rnn/while/Merge_6_grad/tuple/control_dependency_1Identity1gradients/decoder/rnn/while/Merge_6_grad/Switch:1:^gradients/decoder/rnn/while/Merge_6_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/decoder/rnn/while/Switch_6_grad/b_switch*'
_output_shapes
:         
Ў
-gradients/decoder/rnn/while/Enter_2_grad/ExitExitAgradients/decoder/rnn/while/Merge_2_grad/tuple/control_dependency*
T0*
_output_shapes
: 
ф
-gradients/decoder/rnn/while/Enter_3_grad/ExitExitAgradients/decoder/rnn/while/Merge_3_grad/tuple/control_dependency*'
_output_shapes
:         *
T0
ф
-gradients/decoder/rnn/while/Enter_4_grad/ExitExitAgradients/decoder/rnn/while/Merge_4_grad/tuple/control_dependency*'
_output_shapes
:         *
T0
ф
-gradients/decoder/rnn/while/Enter_5_grad/ExitExitAgradients/decoder/rnn/while/Merge_5_grad/tuple/control_dependency*'
_output_shapes
:         *
T0
ф
-gradients/decoder/rnn/while/Enter_6_grad/ExitExitAgradients/decoder/rnn/while/Merge_6_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
T
gradients/zerosConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Е
fgradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lgradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterCgradients/decoder/rnn/while/Merge_2_grad/tuple/control_dependency_1*D
_class:
86loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2*
source	gradients*
_output_shapes

:: 
Р
lgradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterdecoder/rnn/TensorArray*
T0*D
_class:
86loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
Ѓ
bgradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityCgradients/decoder/rnn/while/Merge_2_grad/tuple/control_dependency_1g^gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*D
_class:
86loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2
л
Vgradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3fgradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3agradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2bgradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*'
_output_shapes
:         
п
\gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*/
_class%
#!loc:@decoder/rnn/while/Identity_1*
valueB :
         *
dtype0*
_output_shapes
: 
х
\gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2\gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*/
_class%
#!loc:@decoder/rnn/while/Identity_1*

stack_name *
_output_shapes
:*
	elem_type0
К
\gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnter\gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
▒
bgradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2\gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterdecoder/rnn/while/Identity_1^gradients/Add*
T0*
_output_shapes
: *
swap_memory( 
Љ
agradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2ggradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub*
_output_shapes
: *
	elem_type0
▄
ggradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnter\gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0*
is_constant(
и.
]gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerb^gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2b^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/StackPopV2`^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/StackPopV2N^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/StackPopV2`^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPopV2l^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2n^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1j^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2^^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPopV2`^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPopV2_1l^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2n^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1Z^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPopV2\^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPopV2l^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2n^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1Z^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPopV2\^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPopV2j^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2l^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1X^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPopV2Z^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPopV2b^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/StackPopV2P^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/StackPopV2R^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/StackPopV2`^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/StackPopV2N^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/StackPopV2P^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/StackPopV2b^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/StackPopV2P^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/StackPopV2b^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgs/StackPopV2`^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/StackPopV2N^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/StackPopV2`^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPopV2l^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2n^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1j^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2^^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPopV2`^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPopV2_1l^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2n^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1Z^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPopV2\^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPopV2l^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2n^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1Z^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPopV2\^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPopV2j^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2l^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1X^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPopV2Z^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPopV2b^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/StackPopV2P^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/StackPopV2R^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/StackPopV2b^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgs/StackPopV2P^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul/StackPopV2R^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul_1/StackPopV2`^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/StackPopV2N^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/StackPopV2P^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/StackPopV2
Ч
Ugradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpD^gradients/decoder/rnn/while/Merge_2_grad/tuple/control_dependency_1W^gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
Х
]gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentityVgradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3V^gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*'
_output_shapes
:         
­
_gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentityCgradients/decoder/rnn/while/Merge_2_grad/tuple/control_dependency_1V^gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
_output_shapes
: *
T0*E
_class;
97loc:@gradients/decoder/rnn/while/Switch_2_grad/b_switch
│
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/ShapeShape/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div*
out_type0*
_output_shapes
:*
T0
Д
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Shape_1Const^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"      
М
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/StackPopV2Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
■
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/ConstConst*
_output_shapes
: *W
_classM
KIloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Shape*
valueB :
         *
dtype0
┘
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/f_accStackV2Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/Const*W
_classM
KIloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
├
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/EnterEnterZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
┘
`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/EnterDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Shape^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
Љ
_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
п
egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
Ј
Bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/MulMulCgradients/decoder/rnn/while/Merge_4_grad/tuple/control_dependency_1Mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/StackPopV2*
T0*'
_output_shapes
:         
┘
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/ConstConst*D
_class:
86loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/Floor*
valueB :
         *
dtype0*
_output_shapes
: 
б
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/f_accStackV2Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/Const*D
_class:
86loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/Floor*

stack_name *
_output_shapes
:*
	elem_type0
Ъ
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/EnterEnterHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
д
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/StackPushV2StackPushV2Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/Enter1decoder/rnn/while/rnn/multi_rnn_cell/cell_0/Floor^gradients/Add*
T0*
_output_shapes

:*
swap_memory( 
ы
Mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/StackPopV2
StackPopV2Sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/StackPopV2/Enter^gradients/Sub*
_output_shapes

:*
	elem_type0
┤
Sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/StackPopV2/EnterEnterHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/f_acc*
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
Б
Bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/SumSumBgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/MulTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Х
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/ReshapeReshapeBgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Sum_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
Њ
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1MulOgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/StackPopV2Cgradients/decoder/rnn/while/Merge_4_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
┘
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/ConstConst*B
_class8
64loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div*
valueB :
         *
dtype0*
_output_shapes
: 
ц
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/f_accStackV2Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/Const*B
_class8
64loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div*

stack_name *
_output_shapes
:*
	elem_type0
Б
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/EnterEnterJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
▒
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/StackPushV2StackPushV2Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/Enter/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div^gradients/Add*
T0*'
_output_shapes
:         *
swap_memory( 
■
Ogradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/StackPopV2
StackPopV2Ugradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
И
Ugradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/StackPopV2/EnterEnterJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0
Е
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Sum_1SumDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ў
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Reshape_1ReshapeDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Sum_1Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Shape_1*
_output_shapes

:*
T0*
Tshape0
в
Ogradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/tuple/group_depsNoOpG^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/ReshapeI^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Reshape_1
і
Wgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/tuple/control_dependencyIdentityFgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/ReshapeP^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Reshape*'
_output_shapes
:         
Є
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/tuple/control_dependency_1IdentityHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Reshape_1P^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Reshape_1*
_output_shapes

:
Ђ
.gradients/encoder/rnn/while/Exit_3_grad/b_exitEnter-gradients/decoder/rnn/while/Enter_3_grad/Exit*
is_constant( *
parallel_iterations *'
_output_shapes
:         *9

frame_name+)gradients/encoder/rnn/while/while_context*
T0
Ђ
.gradients/encoder/rnn/while/Exit_4_grad/b_exitEnter-gradients/decoder/rnn/while/Enter_4_grad/Exit*
T0*
is_constant( *
parallel_iterations *'
_output_shapes
:         *9

frame_name+)gradients/encoder/rnn/while/while_context
Ђ
.gradients/encoder/rnn/while/Exit_5_grad/b_exitEnter-gradients/decoder/rnn/while/Enter_5_grad/Exit*
T0*
is_constant( *
parallel_iterations *'
_output_shapes
:         *9

frame_name+)gradients/encoder/rnn/while/while_context
Ђ
.gradients/encoder/rnn/while/Exit_6_grad/b_exitEnter-gradients/decoder/rnn/while/Enter_6_grad/Exit*
T0*
is_constant( *
parallel_iterations *'
_output_shapes
:         *9

frame_name+)gradients/encoder/rnn/while/while_context
м
.gradients/encoder/rnn/while/Exit_2_grad/b_exitEntergradients/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *9

frame_name+)gradients/encoder/rnn/while/while_context
и
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/ShapeShape1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1*
T0*
out_type0*
_output_shapes
:
Е
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Shape_1Const^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
┘
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsagradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/StackPopV2Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ѓ
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/ConstConst*
_output_shapes
: *Y
_classO
MKloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Shape*
valueB :
         *
dtype0
▀
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/f_accStackV2\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/Const*Y
_classO
MKloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
К
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/EnterEnter\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context*
T0
▀
bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/EnterFgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Shape^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
Ћ
agradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ggradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
▄
ggradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0*
is_constant(
Њ
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/MulMulCgradients/decoder/rnn/while/Merge_6_grad/tuple/control_dependency_1Ogradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/StackPopV2*
T0*'
_output_shapes
:         
П
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/ConstConst*F
_class<
:8loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor_1*
valueB :
         *
dtype0*
_output_shapes
: 
е
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/f_accStackV2Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/Const*F
_class<
:8loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor_1*

stack_name *
_output_shapes
:*
	elem_type0
Б
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/EnterEnterJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
г
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/StackPushV2StackPushV2Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/Enter3decoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor_1^gradients/Add*
_output_shapes

:*
swap_memory( *
T0
ш
Ogradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/StackPopV2
StackPopV2Ugradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub*
_output_shapes

:*
	elem_type0
И
Ugradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/StackPopV2/EnterEnterJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
Е
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/SumSumDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/MulVgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╝
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/ReshapeReshapeDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Sumagradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
Ќ
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1MulQgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/StackPopV2Cgradients/decoder/rnn/while/Merge_6_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
П
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/ConstConst*
_output_shapes
: *D
_class:
86loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1*
valueB :
         *
dtype0
ф
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/f_accStackV2Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*D
_class:
86loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1
Д
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/EnterEnterLgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
и
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/StackPushV2StackPushV2Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/Enter1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1^gradients/Add*
T0*'
_output_shapes
:         *
swap_memory( 
ѓ
Qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/StackPopV2
StackPopV2Wgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
╝
Wgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/StackPopV2/EnterEnterLgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
»
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Sum_1SumFgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ъ
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Reshape_1ReshapeFgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Sum_1Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
ы
Qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/tuple/group_depsNoOpI^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/ReshapeK^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Reshape_1
њ
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/tuple/control_dependencyIdentityHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/ReshapeR^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/tuple/group_deps*'
_output_shapes
:         *
T0*[
_classQ
OMloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Reshape
Ј
[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/tuple/control_dependency_1IdentityJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Reshape_1R^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Reshape_1*
_output_shapes

:
и
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/ShapeShape1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2*
T0*
out_type0*
_output_shapes
:
Е
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Shape_1Const^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
┘
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsagradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgs/StackPopV2Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ѓ
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgs/ConstConst*Y
_classO
MKloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
▀
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgs/f_accStackV2\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgs/Const*Y
_classO
MKloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
К
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgs/EnterEnter\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
▀
bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgs/EnterFgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Shape^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
Ћ
agradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ggradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
▄
ggradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnter\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
Г
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/MulMul]gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyOgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul/StackPopV2*
T0*'
_output_shapes
:         
П
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul/ConstConst*F
_class<
:8loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor_2*
valueB :
         *
dtype0*
_output_shapes
: 
е
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul/f_accStackV2Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul/Const*F
_class<
:8loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor_2*

stack_name *
_output_shapes
:*
	elem_type0
Б
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul/EnterEnterJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
г
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul/StackPushV2StackPushV2Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul/Enter3decoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor_2^gradients/Add*
_output_shapes

:*
swap_memory( *
T0
ш
Ogradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul/StackPopV2
StackPopV2Ugradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub*
_output_shapes

:*
	elem_type0
И
Ugradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul/StackPopV2/EnterEnterJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
Е
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/SumSumDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/MulVgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╝
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/ReshapeReshapeDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Sumagradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
▒
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul_1MulQgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul_1/StackPopV2]gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
П
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul_1/ConstConst*D
_class:
86loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2*
valueB :
         *
dtype0*
_output_shapes
: 
ф
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul_1/f_accStackV2Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*D
_class:
86loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2
Д
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul_1/EnterEnterLgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul_1/f_acc*
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
и
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul_1/StackPushV2StackPushV2Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul_1/Enter1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2^gradients/Add*'
_output_shapes
:         *
swap_memory( *
T0
ѓ
Qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul_1/StackPopV2
StackPopV2Wgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
╝
Wgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul_1/StackPopV2/EnterEnterLgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
»
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Sum_1SumFgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul_1Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ъ
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Reshape_1ReshapeFgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Sum_1Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
ы
Qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/tuple/group_depsNoOpI^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/ReshapeK^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Reshape_1
њ
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/tuple/control_dependencyIdentityHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/ReshapeR^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Reshape*'
_output_shapes
:         
Ј
[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/tuple/control_dependency_1IdentityJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Reshape_1R^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Reshape_1*
_output_shapes

:
┐
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/ShapeShape;decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2*
_output_shapes
:*
T0*
out_type0
Ў
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Shape_1Const^gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
М
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/StackPopV2Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Shape_1*
T0*2
_output_shapes 
:         :         
■
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/ConstConst*W
_classM
KIloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
┘
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/f_accStackV2Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/Const*
_output_shapes
:*
	elem_type0*W
_classM
KIloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Shape*

stack_name 
├
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/EnterEnterZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
┘
`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/EnterDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Shape^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
Љ
_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
п
egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0
ф
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/RealDivRealDivWgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/tuple/control_dependencyLgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/RealDiv/Const*'
_output_shapes
:         *
T0
А
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/RealDiv/ConstConst^gradients/Sub*
valueB
 *33s?*
dtype0*
_output_shapes
: 
Д
Bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/SumSumFgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/RealDivTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Х
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/ReshapeReshapeBgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Sum_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
╩
Bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/NegNegMgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/StackPopV2*'
_output_shapes
:         *
T0
с
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/ConstConst*
dtype0*
_output_shapes
: *N
_classD
B@loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2*
valueB :
         
г
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/f_accStackV2Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/Const*N
_classD
B@loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2*

stack_name *
_output_shapes
:*
	elem_type0
Ъ
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/EnterEnterHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
╣
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/StackPushV2StackPushV2Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/Enter;decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2^gradients/Add*'
_output_shapes
:         *
swap_memory( *
T0
Щ
Mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/StackPopV2
StackPopV2Sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
┤
Sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/StackPopV2/EnterEnterHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
Ќ
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/RealDiv_1RealDivBgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/NegLgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/RealDiv/Const*'
_output_shapes
:         *
T0
Ю
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/RealDiv_2RealDivHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/RealDiv_1Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/RealDiv/Const*'
_output_shapes
:         *
T0
ъ
Bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/mulMulWgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/tuple/control_dependencyHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/RealDiv_2*'
_output_shapes
:         *
T0
Д
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Sum_1SumBgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/mulVgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
љ
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Reshape_1ReshapeDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Sum_1Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
в
Ogradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/tuple/group_depsNoOpG^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/ReshapeI^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Reshape_1
і
Wgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/tuple/control_dependencyIdentityFgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/ReshapeP^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/tuple/group_deps*'
_output_shapes
:         *
T0*Y
_classO
MKloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Reshape
 
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/tuple/control_dependency_1IdentityHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Reshape_1P^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Reshape_1*
_output_shapes
: *
T0
с
2gradients/encoder/rnn/while/Switch_3_grad/b_switchMerge.gradients/encoder/rnn/while/Exit_3_grad/b_exit9gradients/encoder/rnn/while/Switch_3_grad_1/NextIteration*
T0*
N*)
_output_shapes
:         : 
с
2gradients/encoder/rnn/while/Switch_4_grad/b_switchMerge.gradients/encoder/rnn/while/Exit_4_grad/b_exit9gradients/encoder/rnn/while/Switch_4_grad_1/NextIteration*
T0*
N*)
_output_shapes
:         : 
с
2gradients/encoder/rnn/while/Switch_5_grad/b_switchMerge.gradients/encoder/rnn/while/Exit_5_grad/b_exit9gradients/encoder/rnn/while/Switch_5_grad_1/NextIteration*)
_output_shapes
:         : *
T0*
N
с
2gradients/encoder/rnn/while/Switch_6_grad/b_switchMerge.gradients/encoder/rnn/while/Exit_6_grad/b_exit9gradients/encoder/rnn/while/Switch_6_grad_1/NextIteration*
T0*
N*)
_output_shapes
:         : 
┴
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/ShapeShape;decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2*
T0*
out_type0*
_output_shapes
:
Џ
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Shape_1Const^gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
┘
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgsBroadcastGradientArgsagradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/StackPopV2Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ѓ
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/ConstConst*Y
_classO
MKloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
▀
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/f_accStackV2\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/Const*Y
_classO
MKloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
К
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/EnterEnter\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
▀
bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/EnterFgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Shape^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
Ћ
agradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ggradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:
▄
ggradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0
░
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/RealDivRealDivYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/tuple/control_dependencyNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/RealDiv/Const*
T0*'
_output_shapes
:         
Б
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/RealDiv/ConstConst^gradients/Sub*
valueB
 *33s?*
dtype0*
_output_shapes
: 
Г
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/SumSumHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/RealDivVgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╝
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/ReshapeReshapeDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Sumagradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:         *
T0*
Tshape0
╬
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/NegNegOgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/StackPopV2*'
_output_shapes
:         *
T0
т
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/ConstConst*N
_classD
B@loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2*
valueB :
         *
dtype0*
_output_shapes
: 
░
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/f_accStackV2Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/Const*N
_classD
B@loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2*

stack_name *
_output_shapes
:*
	elem_type0
Б
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/EnterEnterJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context*
T0
й
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/StackPushV2StackPushV2Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/Enter;decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2^gradients/Add*'
_output_shapes
:         *
swap_memory( *
T0
■
Ogradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/StackPopV2
StackPopV2Ugradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
И
Ugradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/StackPopV2/EnterEnterJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
Ю
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/RealDiv_1RealDivDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/NegNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/RealDiv/Const*
T0*'
_output_shapes
:         
Б
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/RealDiv_2RealDivJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/RealDiv_1Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/RealDiv/Const*
T0*'
_output_shapes
:         
ц
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/mulMulYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/tuple/control_dependencyJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/RealDiv_2*
T0*'
_output_shapes
:         
Г
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Sum_1SumDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/mulXgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ќ
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Reshape_1ReshapeFgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Sum_1Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
ы
Qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/tuple/group_depsNoOpI^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/ReshapeK^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Reshape_1
њ
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/tuple/control_dependencyIdentityHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/ReshapeR^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Reshape*'
_output_shapes
:         
Є
[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/tuple/control_dependency_1IdentityJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Reshape_1R^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Reshape_1*
_output_shapes
: 
┴
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/ShapeShape;decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2*
_output_shapes
:*
T0*
out_type0
Џ
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/Shape_1Const^gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
┘
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgsBroadcastGradientArgsagradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgs/StackPopV2Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ѓ
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgs/ConstConst*Y
_classO
MKloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
▀
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgs/f_accStackV2\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgs/Const*Y
_classO
MKloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
К
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgs/EnterEnter\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
▀
bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgs/EnterFgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/Shape^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
Ћ
agradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ggradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
▄
ggradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnter\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
░
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/RealDivRealDivYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/tuple/control_dependencyNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/RealDiv/Const*
T0*'
_output_shapes
:         
Б
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/RealDiv/ConstConst^gradients/Sub*
valueB
 *33s?*
dtype0*
_output_shapes
: 
Г
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/SumSumHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/RealDivVgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╝
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/ReshapeReshapeDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/Sumagradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
╬
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/NegNegOgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/StackPopV2*
T0*'
_output_shapes
:         
Ю
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/RealDiv_1RealDivDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/NegNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/RealDiv/Const*'
_output_shapes
:         *
T0
Б
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/RealDiv_2RealDivJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/RealDiv_1Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/RealDiv/Const*
T0*'
_output_shapes
:         
ц
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/mulMulYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/tuple/control_dependencyJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/RealDiv_2*'
_output_shapes
:         *
T0
Г
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/Sum_1SumDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/mulXgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ќ
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/Reshape_1ReshapeFgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/Sum_1Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
ы
Qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/tuple/group_depsNoOpI^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/ReshapeK^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/Reshape_1
њ
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/tuple/control_dependencyIdentityHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/ReshapeR^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/Reshape*'
_output_shapes
:         
Є
[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/tuple/control_dependency_1IdentityJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/Reshape_1R^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/Reshape_1*
_output_shapes
: 
╠
9gradients/decoder/rnn/while/Switch_2_grad_1/NextIterationNextIteration_gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
ј
/gradients/encoder/rnn/while/Merge_3_grad/SwitchSwitch2gradients/encoder/rnn/while/Switch_3_grad/b_switchgradients/b_count_6*
T0*E
_class;
97loc:@gradients/encoder/rnn/while/Switch_3_grad/b_switch*:
_output_shapes(
&:         :         
s
9gradients/encoder/rnn/while/Merge_3_grad/tuple/group_depsNoOp0^gradients/encoder/rnn/while/Merge_3_grad/Switch
│
Agradients/encoder/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity/gradients/encoder/rnn/while/Merge_3_grad/Switch:^gradients/encoder/rnn/while/Merge_3_grad/tuple/group_deps*'
_output_shapes
:         *
T0*E
_class;
97loc:@gradients/encoder/rnn/while/Switch_3_grad/b_switch
и
Cgradients/encoder/rnn/while/Merge_3_grad/tuple/control_dependency_1Identity1gradients/encoder/rnn/while/Merge_3_grad/Switch:1:^gradients/encoder/rnn/while/Merge_3_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/encoder/rnn/while/Switch_3_grad/b_switch*'
_output_shapes
:         
ј
/gradients/encoder/rnn/while/Merge_4_grad/SwitchSwitch2gradients/encoder/rnn/while/Switch_4_grad/b_switchgradients/b_count_6*:
_output_shapes(
&:         :         *
T0*E
_class;
97loc:@gradients/encoder/rnn/while/Switch_4_grad/b_switch
s
9gradients/encoder/rnn/while/Merge_4_grad/tuple/group_depsNoOp0^gradients/encoder/rnn/while/Merge_4_grad/Switch
│
Agradients/encoder/rnn/while/Merge_4_grad/tuple/control_dependencyIdentity/gradients/encoder/rnn/while/Merge_4_grad/Switch:^gradients/encoder/rnn/while/Merge_4_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/encoder/rnn/while/Switch_4_grad/b_switch*'
_output_shapes
:         
и
Cgradients/encoder/rnn/while/Merge_4_grad/tuple/control_dependency_1Identity1gradients/encoder/rnn/while/Merge_4_grad/Switch:1:^gradients/encoder/rnn/while/Merge_4_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/encoder/rnn/while/Switch_4_grad/b_switch*'
_output_shapes
:         
ј
/gradients/encoder/rnn/while/Merge_5_grad/SwitchSwitch2gradients/encoder/rnn/while/Switch_5_grad/b_switchgradients/b_count_6*
T0*E
_class;
97loc:@gradients/encoder/rnn/while/Switch_5_grad/b_switch*:
_output_shapes(
&:         :         
s
9gradients/encoder/rnn/while/Merge_5_grad/tuple/group_depsNoOp0^gradients/encoder/rnn/while/Merge_5_grad/Switch
│
Agradients/encoder/rnn/while/Merge_5_grad/tuple/control_dependencyIdentity/gradients/encoder/rnn/while/Merge_5_grad/Switch:^gradients/encoder/rnn/while/Merge_5_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/encoder/rnn/while/Switch_5_grad/b_switch*'
_output_shapes
:         
и
Cgradients/encoder/rnn/while/Merge_5_grad/tuple/control_dependency_1Identity1gradients/encoder/rnn/while/Merge_5_grad/Switch:1:^gradients/encoder/rnn/while/Merge_5_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/encoder/rnn/while/Switch_5_grad/b_switch*'
_output_shapes
:         
ј
/gradients/encoder/rnn/while/Merge_6_grad/SwitchSwitch2gradients/encoder/rnn/while/Switch_6_grad/b_switchgradients/b_count_6*:
_output_shapes(
&:         :         *
T0*E
_class;
97loc:@gradients/encoder/rnn/while/Switch_6_grad/b_switch
s
9gradients/encoder/rnn/while/Merge_6_grad/tuple/group_depsNoOp0^gradients/encoder/rnn/while/Merge_6_grad/Switch
│
Agradients/encoder/rnn/while/Merge_6_grad/tuple/control_dependencyIdentity/gradients/encoder/rnn/while/Merge_6_grad/Switch:^gradients/encoder/rnn/while/Merge_6_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/encoder/rnn/while/Switch_6_grad/b_switch*'
_output_shapes
:         
и
Cgradients/encoder/rnn/while/Merge_6_grad/tuple/control_dependency_1Identity1gradients/encoder/rnn/while/Merge_6_grad/Switch:1:^gradients/encoder/rnn/while/Merge_6_grad/tuple/group_deps*'
_output_shapes
:         *
T0*E
_class;
97loc:@gradients/encoder/rnn/while/Switch_6_grad/b_switch
С
gradients/AddNAddNYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/tuple/control_dependencyYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/tuple/control_dependency*
T0*[
_classQ
OMloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Reshape*
N*'
_output_shapes
:         
¤
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ShapeShape?decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2*
_output_shapes
:*
T0*
out_type0
╬
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1Shape<decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
_output_shapes
:*
T0*
out_type0
њ
`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgskgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
ќ
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/ConstConst*c
_classY
WUloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
§
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_accStackV2fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const*c
_classY
WUloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
█
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterEnterfgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
§
lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterPgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
Е
kgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
­
qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterfgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
џ
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1*
valueB :
         
Ѓ
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1*
	elem_type0*e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1*

stack_name *
_output_shapes
:
▀
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Enterhgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
Ѓ
ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
Г
mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:
З
sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterhgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
Ы
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/MulMulgradients/AddNYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPopV2*
T0*'
_output_shapes
:         
­
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/ConstConst*O
_classE
CAloc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
valueB :
         *
dtype0*
_output_shapes
: 
┼
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/f_accStackV2Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/Const*O
_classE
CAloc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*

stack_name *
_output_shapes
:*
	elem_type0
и
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/EnterEnterTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
м
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPushV2StackPushV2Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/Enter<decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1^gradients/Add*
T0*'
_output_shapes
:         *
swap_memory( 
њ
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPopV2
StackPopV2_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
╠
_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPopV2/EnterEnterTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/f_acc*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0*
is_constant(
К
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/SumSumNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
┌
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ReshapeReshapeNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Sumkgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
Ш
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1Mul[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPopV2gradients/AddN*
T0*'
_output_shapes
:         
ш
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/ConstConst*R
_classH
FDloc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2*
valueB :
         *
dtype0*
_output_shapes
: 
╠
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/f_accStackV2Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/Const*
	elem_type0*R
_classH
FDloc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2*

stack_name *
_output_shapes
:
╗
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/EnterEnterVgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context*
T0
┘
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPushV2StackPushV2Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/Enter?decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2^gradients/Add*
T0*'
_output_shapes
:         *
swap_memory( 
ќ
[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPopV2
StackPopV2agradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*'
_output_shapes
:         
л
agradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPopV2/EnterEnterVgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
═
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Sum_1SumPgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Я
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape_1ReshapePgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Sum_1mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*'
_output_shapes
:         *
T0*
Tshape0
Ј
[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/tuple/group_depsNoOpS^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ReshapeU^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape_1
║
cgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/tuple/control_dependencyIdentityRgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape\^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape*'
_output_shapes
:         
└
egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/tuple/control_dependency_1IdentityTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape_1\^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape_1*'
_output_shapes
:         
ф
-gradients/encoder/rnn/while/Enter_3_grad/ExitExitAgradients/encoder/rnn/while/Merge_3_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
ф
-gradients/encoder/rnn/while/Enter_4_grad/ExitExitAgradients/encoder/rnn/while/Merge_4_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
ф
-gradients/encoder/rnn/while/Enter_5_grad/ExitExitAgradients/encoder/rnn/while/Merge_5_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
ф
-gradients/encoder/rnn/while/Enter_6_grad/ExitExitAgradients/encoder/rnn/while/Merge_6_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
П
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPopV2cgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
н
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1_grad/TanhGradTanhGradYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPopV2egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/tuple/control_dependency_1*'
_output_shapes
:         *
T0
│
Dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/ShapeShape/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div*
T0*
out_type0*
_output_shapes
:
Е
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Shape_1Const^gradients/Sub_1*
valueB"      *
dtype0*
_output_shapes
:
М
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/StackPopV2Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
■
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/ConstConst*W
_classM
KIloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
┘
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/f_accStackV2Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*W
_classM
KIloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Shape
├
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/EnterEnterZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
█
`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/EnterDgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Shape^gradients/Add_1*
T0*
_output_shapes
:*
swap_memory( 
Њ
_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
п
egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
с*
[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/b_syncControlTriggerb^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/StackPopV2`^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/StackPopV2N^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/StackPopV2`^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPopV2l^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2n^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1j^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2^^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPopV2`^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPopV2_1l^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2n^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1Z^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPopV2\^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPopV2l^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2n^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1Z^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPopV2\^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPopV2j^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2l^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1X^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPopV2Z^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPopV2b^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/StackPopV2P^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/StackPopV2R^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/StackPopV2`^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/StackPopV2N^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/StackPopV2P^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/StackPopV2b^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/StackPopV2P^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/StackPopV2`^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/StackPopV2N^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/StackPopV2`^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPopV2l^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2n^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1j^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2^^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPopV2`^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPopV2_1l^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2n^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1Z^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPopV2\^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPopV2l^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2n^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1Z^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPopV2\^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPopV2j^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2l^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1X^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPopV2Z^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPopV2b^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/StackPopV2P^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/StackPopV2R^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/StackPopV2`^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/StackPopV2N^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/StackPopV2P^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/StackPopV2
Ј
Bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/MulMulCgradients/encoder/rnn/while/Merge_4_grad/tuple/control_dependency_1Mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/StackPopV2*'
_output_shapes
:         *
T0
┘
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/ConstConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/Floor*
valueB :
         
б
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/f_accStackV2Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/Const*D
_class:
86loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/Floor*

stack_name *
_output_shapes
:*
	elem_type0
Ъ
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/EnterEnterHgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
е
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/StackPushV2StackPushV2Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/Enter1encoder/rnn/while/rnn/multi_rnn_cell/cell_0/Floor^gradients/Add_1*
T0*
_output_shapes

:*
swap_memory( 
з
Mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/StackPopV2
StackPopV2Sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/StackPopV2/Enter^gradients/Sub_1*
_output_shapes

:*
	elem_type0
┤
Sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/StackPopV2/EnterEnterHgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0
Б
Bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/SumSumBgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/MulTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Х
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/ReshapeReshapeBgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Sum_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/StackPopV2*
Tshape0*'
_output_shapes
:         *
T0
Њ
Dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1MulOgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/StackPopV2Cgradients/encoder/rnn/while/Merge_4_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
┘
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *B
_class8
64loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div*
valueB :
         
ц
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/f_accStackV2Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*B
_class8
64loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div
Б
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/EnterEnterJgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
│
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/StackPushV2StackPushV2Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/Enter/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div^gradients/Add_1*
T0*'
_output_shapes
:         *
swap_memory( 
ђ
Ogradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/StackPopV2
StackPopV2Ugradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*'
_output_shapes
:         *
	elem_type0
И
Ugradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/StackPopV2/EnterEnterJgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0
Е
Dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Sum_1SumDgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ў
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Reshape_1ReshapeDgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Sum_1Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
в
Ogradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/tuple/group_depsNoOpG^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/ReshapeI^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Reshape_1
і
Wgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/tuple/control_dependencyIdentityFgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/ReshapeP^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Reshape*'
_output_shapes
:         
Є
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/tuple/control_dependency_1IdentityHgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Reshape_1P^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Reshape_1*
_output_shapes

:
и
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/ShapeShape1encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Shape_1Const^gradients/Sub_1*
valueB"      *
dtype0*
_output_shapes
:
┘
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsagradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/StackPopV2Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ѓ
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/ConstConst*
_output_shapes
: *Y
_classO
MKloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Shape*
valueB :
         *
dtype0
▀
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/f_accStackV2\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/Const*Y
_classO
MKloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
К
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/EnterEnter\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
р
bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/EnterFgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Shape^gradients/Add_1*
_output_shapes
:*
swap_memory( *
T0
Ќ
agradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ggradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
▄
ggradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
Њ
Dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/MulMulCgradients/encoder/rnn/while/Merge_6_grad/tuple/control_dependency_1Ogradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/StackPopV2*
T0*'
_output_shapes
:         
П
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/ConstConst*F
_class<
:8loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor_1*
valueB :
         *
dtype0*
_output_shapes
: 
е
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/f_accStackV2Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/Const*
_output_shapes
:*
	elem_type0*F
_class<
:8loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor_1*

stack_name 
Б
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/EnterEnterJgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context*
T0
«
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/StackPushV2StackPushV2Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/Enter3encoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor_1^gradients/Add_1*
T0*
_output_shapes

:*
swap_memory( 
э
Ogradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/StackPopV2
StackPopV2Ugradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub_1*
_output_shapes

:*
	elem_type0
И
Ugradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/StackPopV2/EnterEnterJgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/f_acc*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0*
is_constant(
Е
Dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/SumSumDgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/MulVgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╝
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/ReshapeReshapeDgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Sumagradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
Ќ
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1MulQgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/StackPopV2Cgradients/encoder/rnn/while/Merge_6_grad/tuple/control_dependency_1*'
_output_shapes
:         *
T0
П
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1*
valueB :
         
ф
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/f_accStackV2Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/Const*D
_class:
86loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1*

stack_name *
_output_shapes
:*
	elem_type0
Д
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/EnterEnterLgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
╣
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/StackPushV2StackPushV2Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/Enter1encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1^gradients/Add_1*
T0*'
_output_shapes
:         *
swap_memory( 
ё
Qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/StackPopV2
StackPopV2Wgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*'
_output_shapes
:         
╝
Wgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/StackPopV2/EnterEnterLgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
»
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Sum_1SumFgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ъ
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Reshape_1ReshapeFgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Sum_1Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
ы
Qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/tuple/group_depsNoOpI^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/ReshapeK^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Reshape_1
њ
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/tuple/control_dependencyIdentityHgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/ReshapeR^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Reshape*'
_output_shapes
:         
Ј
[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/tuple/control_dependency_1IdentityJgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Reshape_1R^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Reshape_1*
_output_shapes

:
х
gradients/AddN_1AddNCgradients/decoder/rnn/while/Merge_5_grad/tuple/control_dependency_1Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1_grad/TanhGrad*'
_output_shapes
:         *
T0*E
_class;
97loc:@gradients/decoder/rnn/while/Switch_5_grad/b_switch*
N
╔
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ShapeShape9decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul*
T0*
out_type0*
_output_shapes
:
═
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1Shape;decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1*
T0*
out_type0*
_output_shapes
:
њ
`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgskgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
ќ
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *c
_classY
WUloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape*
valueB :
         
§
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/f_accStackV2fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/Const*c
_classY
WUloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
█
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterEnterfgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
§
lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterPgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
Е
kgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
­
qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterfgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
џ
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1Const*e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
Ѓ
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1*e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
▀
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Enterhgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context*
T0*
is_constant(
Ѓ
ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
Г
mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
З
sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterhgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
Ѕ
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/SumSumgradients/AddN_1`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
┌
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ReshapeReshapeNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Sumkgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
Ї
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_1bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Я
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1ReshapePgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Sum_1mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*'
_output_shapes
:         
Ј
[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/group_depsNoOpS^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ReshapeU^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1
║
cgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/control_dependencyIdentityRgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape\^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/group_deps*e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape*'
_output_shapes
:         *
T0
└
egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/control_dependency_1IdentityTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1\^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1*'
_output_shapes
:         
┐
Dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/ShapeShape;encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2*
T0*
out_type0*
_output_shapes
:
Џ
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Shape_1Const^gradients/Sub_1*
valueB *
dtype0*
_output_shapes
: 
М
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/StackPopV2Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Shape_1*
T0*2
_output_shapes 
:         :         
■
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/ConstConst*W
_classM
KIloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
┘
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/f_accStackV2Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/Const*W
_classM
KIloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
├
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/EnterEnterZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
█
`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/EnterDgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Shape^gradients/Add_1*
T0*
_output_shapes
:*
swap_memory( 
Њ
_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
п
egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
ф
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/RealDivRealDivWgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/tuple/control_dependencyLgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/RealDiv/Const*
T0*'
_output_shapes
:         
Б
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/RealDiv/ConstConst^gradients/Sub_1*
valueB
 *33s?*
dtype0*
_output_shapes
: 
Д
Bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/SumSumFgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/RealDivTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Х
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/ReshapeReshapeBgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Sum_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
╩
Bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/NegNegMgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/StackPopV2*
T0*'
_output_shapes
:         
с
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/ConstConst*
dtype0*
_output_shapes
: *N
_classD
B@loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2*
valueB :
         
г
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/f_accStackV2Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/Const*N
_classD
B@loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2*

stack_name *
_output_shapes
:*
	elem_type0
Ъ
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/EnterEnterHgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
╗
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/StackPushV2StackPushV2Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/Enter;encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2^gradients/Add_1*'
_output_shapes
:         *
swap_memory( *
T0
Ч
Mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/StackPopV2
StackPopV2Sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/StackPopV2/Enter^gradients/Sub_1*'
_output_shapes
:         *
	elem_type0
┤
Sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/StackPopV2/EnterEnterHgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
Ќ
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/RealDiv_1RealDivBgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/NegLgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/RealDiv/Const*
T0*'
_output_shapes
:         
Ю
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/RealDiv_2RealDivHgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/RealDiv_1Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/RealDiv/Const*
T0*'
_output_shapes
:         
ъ
Bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/mulMulWgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/tuple/control_dependencyHgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/RealDiv_2*'
_output_shapes
:         *
T0
Д
Dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Sum_1SumBgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/mulVgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
љ
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Reshape_1ReshapeDgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Sum_1Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
в
Ogradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/tuple/group_depsNoOpG^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/ReshapeI^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Reshape_1
і
Wgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/tuple/control_dependencyIdentityFgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/ReshapeP^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Reshape*'
_output_shapes
:         
 
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/tuple/control_dependency_1IdentityHgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Reshape_1P^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Reshape_1*
_output_shapes
: *
T0
┴
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/ShapeShape;encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2*
T0*
out_type0*
_output_shapes
:
Ю
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Shape_1Const^gradients/Sub_1*
valueB *
dtype0*
_output_shapes
: 
┘
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgsBroadcastGradientArgsagradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/StackPopV2Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
ѓ
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/ConstConst*Y
_classO
MKloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
▀
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/f_accStackV2\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/Const*
_output_shapes
:*
	elem_type0*Y
_classO
MKloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Shape*

stack_name 
К
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/EnterEnter\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
р
bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/EnterFgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Shape^gradients/Add_1*
_output_shapes
:*
swap_memory( *
T0
Ќ
agradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ggradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
▄
ggradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
░
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/RealDivRealDivYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/tuple/control_dependencyNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/RealDiv/Const*
T0*'
_output_shapes
:         
Ц
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/RealDiv/ConstConst^gradients/Sub_1*
_output_shapes
: *
valueB
 *33s?*
dtype0
Г
Dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/SumSumHgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/RealDivVgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╝
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/ReshapeReshapeDgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Sumagradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
╬
Dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/NegNegOgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/StackPopV2*
T0*'
_output_shapes
:         
т
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/ConstConst*
_output_shapes
: *N
_classD
B@loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2*
valueB :
         *
dtype0
░
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/f_accStackV2Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/Const*N
_classD
B@loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2*

stack_name *
_output_shapes
:*
	elem_type0
Б
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/EnterEnterJgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
┐
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/StackPushV2StackPushV2Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/Enter;encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2^gradients/Add_1*
T0*'
_output_shapes
:         *
swap_memory( 
ђ
Ogradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/StackPopV2
StackPopV2Ugradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/StackPopV2/Enter^gradients/Sub_1*'
_output_shapes
:         *
	elem_type0
И
Ugradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/StackPopV2/EnterEnterJgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
Ю
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/RealDiv_1RealDivDgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/NegNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/RealDiv/Const*
T0*'
_output_shapes
:         
Б
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/RealDiv_2RealDivJgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/RealDiv_1Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/RealDiv/Const*'
_output_shapes
:         *
T0
ц
Dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/mulMulYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/tuple/control_dependencyJgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/RealDiv_2*
T0*'
_output_shapes
:         
Г
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Sum_1SumDgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/mulXgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ќ
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Reshape_1ReshapeFgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Sum_1Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
ы
Qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/tuple/group_depsNoOpI^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/ReshapeK^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Reshape_1
њ
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/tuple/control_dependencyIdentityHgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/ReshapeR^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Reshape*'
_output_shapes
:         
Є
[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/tuple/control_dependency_1IdentityJgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Reshape_1R^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Reshape_1*
_output_shapes
: 
╦
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/ShapeShape=decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid*
T0*
out_type0*
_output_shapes
:
г
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1Shapedecoder/rnn/while/Identity_5*
T0*
out_type0*
_output_shapes
:
ї
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsigradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2kgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
њ
dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/ConstConst*a
_classW
USloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
э
dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStackV2dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/Const*a
_classW
USloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
О
dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/EnterEnterdgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context*
T0*
is_constant(
э
jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/EnterNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
Ц
igradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ogradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
В
ogradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterdgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
ќ
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1Const*c
_classY
WUloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
§
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1StackV2fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1*
	elem_type0*c
_classY
WUloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1*

stack_name *
_output_shapes
:
█
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Enterfgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
§
lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
Е
kgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
­
qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterfgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
├
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/MulMulcgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/control_dependencyWgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPopV2*'
_output_shapes
:         *
T0
╬
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/ConstConst*/
_class%
#!loc:@decoder/rnn/while/Identity_5*
valueB :
         *
dtype0*
_output_shapes
: 
А
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/f_accStackV2Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/Const*/
_class%
#!loc:@decoder/rnn/while/Identity_5*

stack_name *
_output_shapes
:*
	elem_type0
│
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/EnterEnterRgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
«
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPushV2StackPushV2Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/Enterdecoder/rnn/while/Identity_5^gradients/Add*
T0*'
_output_shapes
:         *
swap_memory( 
ј
Wgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPopV2
StackPopV2]gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
╚
]gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPopV2/EnterEnterRgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/f_acc*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0*
is_constant(
┴
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/SumSumLgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
н
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/ReshapeReshapeLgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Sumigradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
К
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1MulYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPopV2cgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
ы
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/ConstConst*P
_classF
DBloc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid*
valueB :
         *
dtype0*
_output_shapes
: 
к
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/f_accStackV2Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*P
_classF
DBloc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid
и
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/EnterEnterTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/f_acc*
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
М
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPushV2StackPushV2Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/Enter=decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid^gradients/Add*
T0*'
_output_shapes
:         *
swap_memory( 
њ
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPopV2
StackPopV2_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
╠
_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPopV2/EnterEnterTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/f_acc*
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
К
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Sum_1SumNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
┌
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape_1ReshapeNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Sum_1kgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*'
_output_shapes
:         
Ѕ
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/tuple/group_depsNoOpQ^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/ReshapeS^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape_1
▓
agradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/tuple/control_dependencyIdentityPgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/ReshapeZ^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/tuple/group_deps*c
_classY
WUloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape*'
_output_shapes
:         *
T0
И
cgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/tuple/control_dependency_1IdentityRgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape_1Z^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/tuple/group_deps*'
_output_shapes
:         *
T0*e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape_1
¤
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ShapeShape?decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1*
T0*
out_type0*
_output_shapes
:
╠
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1Shape:decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh*
T0*
out_type0*
_output_shapes
:
њ
`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgskgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
ќ
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/ConstConst*c
_classY
WUloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
§
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_accStackV2fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const*c
_classY
WUloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
█
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterEnterfgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
§
lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterPgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
Е
kgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
­
qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterfgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
џ
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1Const*
_output_shapes
: *e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1*
valueB :
         *
dtype0
Ѓ
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1*e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
▀
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Enterhgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
Ѓ
ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
Г
mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
З
sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterhgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
╔
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/MulMulegradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/control_dependency_1Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPopV2*
T0*'
_output_shapes
:         
Ь
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/ConstConst*M
_classC
A?loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh*
valueB :
         *
dtype0*
_output_shapes
: 
├
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/f_accStackV2Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/Const*M
_classC
A?loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh*

stack_name *
_output_shapes
:*
	elem_type0
и
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/EnterEnterTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/f_acc*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context*
T0*
is_constant(
л
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPushV2StackPushV2Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/Enter:decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh^gradients/Add*
T0*'
_output_shapes
:         *
swap_memory( 
њ
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPopV2
StackPopV2_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
╠
_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPopV2/EnterEnterTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/f_acc*
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
К
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/SumSumNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
┌
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ReshapeReshapeNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Sumkgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
═
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1Mul[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPopV2egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
ш
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/ConstConst*R
_classH
FDloc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1*
valueB :
         *
dtype0*
_output_shapes
: 
╠
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/f_accStackV2Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/Const*R
_classH
FDloc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:*
	elem_type0
╗
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/EnterEnterVgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
┘
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPushV2StackPushV2Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/Enter?decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1^gradients/Add*'
_output_shapes
:         *
swap_memory( *
T0
ќ
[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPopV2
StackPopV2agradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
л
agradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPopV2/EnterEnterVgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
═
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Sum_1SumPgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Я
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape_1ReshapePgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Sum_1mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*'
_output_shapes
:         
Ј
[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/tuple/group_depsNoOpS^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ReshapeU^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape_1
║
cgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/tuple/control_dependencyIdentityRgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape\^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape*'
_output_shapes
:         
└
egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/tuple/control_dependency_1IdentityTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape_1\^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape_1*'
_output_shapes
:         
¤
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ShapeShape?encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2*
T0*
out_type0*
_output_shapes
:
╬
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1Shape<encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
out_type0*
_output_shapes
:*
T0
њ
`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgskgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
ќ
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/ConstConst*
_output_shapes
: *c
_classY
WUloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape*
valueB :
         *
dtype0
§
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_accStackV2fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*c
_classY
WUloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape
█
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterEnterfgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
 
lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterPgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape^gradients/Add_1*
T0*
_output_shapes
:*
swap_memory( 
Ф
kgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:
­
qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterfgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
џ
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1*
valueB :
         
Ѓ
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1*
_output_shapes
:*
	elem_type0*e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1*

stack_name 
▀
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Enterhgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
Ё
ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1^gradients/Add_1*
_output_shapes
:*
swap_memory( *
T0
»
mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
З
sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterhgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
й
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/MulMulYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/tuple/control_dependencyYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPopV2*'
_output_shapes
:         *
T0
­
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/ConstConst*
dtype0*
_output_shapes
: *O
_classE
CAloc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
valueB :
         
┼
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/f_accStackV2Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/Const*
	elem_type0*O
_classE
CAloc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*

stack_name *
_output_shapes
:
и
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/EnterEnterTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
н
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPushV2StackPushV2Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/Enter<encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1^gradients/Add_1*
T0*'
_output_shapes
:         *
swap_memory( 
ћ
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPopV2
StackPopV2_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub_1*'
_output_shapes
:         *
	elem_type0
╠
_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPopV2/EnterEnterTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/f_acc*
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
К
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/SumSumNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
┌
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ReshapeReshapeNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Sumkgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
┴
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1Mul[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPopV2Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/tuple/control_dependency*'
_output_shapes
:         *
T0
ш
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/ConstConst*
_output_shapes
: *R
_classH
FDloc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2*
valueB :
         *
dtype0
╠
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/f_accStackV2Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*R
_classH
FDloc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2
╗
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/EnterEnterVgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
█
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPushV2StackPushV2Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/Enter?encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2^gradients/Add_1*
T0*'
_output_shapes
:         *
swap_memory( 
ў
[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPopV2
StackPopV2agradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*'
_output_shapes
:         *
	elem_type0
л
agradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPopV2/EnterEnterVgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
═
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Sum_1SumPgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Я
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape_1ReshapePgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Sum_1mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*'
_output_shapes
:         *
T0*
Tshape0
Ј
[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/tuple/group_depsNoOpS^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/ReshapeU^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape_1
║
cgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/tuple/control_dependencyIdentityRgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape\^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape*'
_output_shapes
:         
└
egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/tuple/control_dependency_1IdentityTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape_1\^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Reshape_1*'
_output_shapes
:         
О
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPopV2agradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/tuple/control_dependency*'
_output_shapes
:         *
T0
П
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGrad[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPopV2cgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
м
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_grad/TanhGradTanhGradYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPopV2egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/tuple/control_dependency_1*'
_output_shapes
:         *
T0
П
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPopV2cgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
н
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1_grad/TanhGradTanhGradYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPopV2egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
╦
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/ShapeShape=decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:2*
T0*
out_type0*
_output_shapes
:
Б
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1Const^gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
ы
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsigradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
њ
dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *a
_classW
USloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape*
valueB :
         
э
dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/f_accStackV2dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/Const*a
_classW
USloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
О
dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/EnterEnterdgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
э
jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/EnterNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
Ц
igradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ogradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
В
ogradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnterdgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0
═
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/SumSumXgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGrad^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
н
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/ReshapeReshapeLgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Sumigradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
Л
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Sum_1SumXgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGrad`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
«
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Reshape_1ReshapeNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Sum_1Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
Ѕ
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/tuple/group_depsNoOpQ^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/ReshapeS^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Reshape_1
▓
agradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/tuple/control_dependencyIdentityPgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/ReshapeZ^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Reshape*'
_output_shapes
:         
Д
cgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/tuple/control_dependency_1IdentityRgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Reshape_1Z^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/tuple/group_deps*
_output_shapes
: *
T0*e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Reshape_1
р
9gradients/decoder/rnn/while/Switch_5_grad_1/NextIterationNextIterationcgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
х
gradients/AddN_2AddNCgradients/encoder/rnn/while/Merge_5_grad/tuple/control_dependency_1Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1_grad/TanhGrad*
T0*E
_class;
97loc:@gradients/encoder/rnn/while/Switch_5_grad/b_switch*
N*'
_output_shapes
:         
╔
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ShapeShape9encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul*
_output_shapes
:*
T0*
out_type0
═
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1Shape;encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1*
T0*
out_type0*
_output_shapes
:
њ
`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgskgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:         :         *
T0
ќ
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/ConstConst*c
_classY
WUloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
§
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/f_accStackV2fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/Const*c
_classY
WUloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
█
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterEnterfgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context*
T0*
is_constant(
 
lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterPgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape^gradients/Add_1*
T0*
_output_shapes
:*
swap_memory( 
Ф
kgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
­
qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterfgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
џ
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1Const*e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
Ѓ
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1*e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
▀
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Enterhgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
Ё
ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1^gradients/Add_1*
_output_shapes
:*
swap_memory( *
T0
»
mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
З
sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterhgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
Ѕ
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/SumSumgradients/AddN_2`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
┌
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ReshapeReshapeNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Sumkgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:         *
T0*
Tshape0
Ї
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_2bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Я
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1ReshapePgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Sum_1mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*'
_output_shapes
:         
Ј
[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/group_depsNoOpS^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/ReshapeU^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1
║
cgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/control_dependencyIdentityRgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape\^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/group_deps*e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape*'
_output_shapes
:         *
T0
└
egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/control_dependency_1IdentityTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1\^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Reshape_1*'
_output_shapes
:         
В
Qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_grad/concatConcatV2Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1_grad/SigmoidGradRgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_grad/TanhGradagradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/tuple/control_dependencyZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2_grad/SigmoidGradWgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/Const*
T0*
N*'
_output_shapes
:         *

Tidx0
Е
Wgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/ConstConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
╦
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/ShapeShape=encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid*
T0*
out_type0*
_output_shapes
:
г
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1Shapeencoder/rnn/while/Identity_5*
_output_shapes
:*
T0*
out_type0
ї
^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsigradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2kgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
њ
dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *a
_classW
USloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape*
valueB :
         
э
dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStackV2dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*a
_classW
USloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape
О
dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/EnterEnterdgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
щ
jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/EnterNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape^gradients/Add_1*
T0*
_output_shapes
:*
swap_memory( 
Д
igradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ogradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
В
ogradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterdgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0
ќ
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1Const*c
_classY
WUloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
§
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1StackV2fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1*c
_classY
WUloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
█
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Enterfgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
 
lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1^gradients/Add_1*
T0*
_output_shapes
:*
swap_memory( 
Ф
kgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
­
qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterfgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
├
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/MulMulcgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/control_dependencyWgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPopV2*'
_output_shapes
:         *
T0
╬
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/ConstConst*/
_class%
#!loc:@encoder/rnn/while/Identity_5*
valueB :
         *
dtype0*
_output_shapes
: 
А
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/f_accStackV2Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/Const*

stack_name *
_output_shapes
:*
	elem_type0*/
_class%
#!loc:@encoder/rnn/while/Identity_5
│
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/EnterEnterRgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context*
T0
░
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPushV2StackPushV2Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/Enterencoder/rnn/while/Identity_5^gradients/Add_1*'
_output_shapes
:         *
swap_memory( *
T0
љ
Wgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPopV2
StackPopV2]gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPopV2/Enter^gradients/Sub_1*'
_output_shapes
:         *
	elem_type0
╚
]gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPopV2/EnterEnterRgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
┴
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/SumSumLgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
н
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/ReshapeReshapeLgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Sumigradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:         *
T0*
Tshape0
К
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1MulYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPopV2cgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
ы
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/ConstConst*P
_classF
DBloc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid*
valueB :
         *
dtype0*
_output_shapes
: 
к
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/f_accStackV2Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/Const*P
_classF
DBloc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid*

stack_name *
_output_shapes
:*
	elem_type0
и
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/EnterEnterTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context*
T0
Н
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPushV2StackPushV2Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/Enter=encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid^gradients/Add_1*'
_output_shapes
:         *
swap_memory( *
T0
ћ
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPopV2
StackPopV2_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*'
_output_shapes
:         *
	elem_type0
╠
_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPopV2/EnterEnterTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
К
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Sum_1SumNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
┌
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape_1ReshapeNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Sum_1kgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*'
_output_shapes
:         *
T0*
Tshape0
Ѕ
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/tuple/group_depsNoOpQ^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/ReshapeS^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape_1
▓
agradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/tuple/control_dependencyIdentityPgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/ReshapeZ^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/tuple/group_deps*'
_output_shapes
:         *
T0*c
_classY
WUloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape
И
cgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/tuple/control_dependency_1IdentityRgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape_1Z^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/tuple/group_deps*e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Reshape_1*'
_output_shapes
:         *
T0
¤
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ShapeShape?encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1*
out_type0*
_output_shapes
:*
T0
╠
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1Shape:encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh*
T0*
out_type0*
_output_shapes
:
њ
`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgskgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
ќ
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/ConstConst*c
_classY
WUloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
§
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_accStackV2fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const*c
_classY
WUloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
█
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterEnterfgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
 
lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterPgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape^gradients/Add_1*
_output_shapes
:*
swap_memory( *
T0
Ф
kgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
­
qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterfgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
џ
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1*
valueB :
         
Ѓ
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1*e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
▀
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Enterhgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
Ё
ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1^gradients/Add_1*
_output_shapes
:*
swap_memory( *
T0
»
mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
З
sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterhgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
╔
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/MulMulegradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/control_dependency_1Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPopV2*'
_output_shapes
:         *
T0
Ь
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/ConstConst*M
_classC
A?loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh*
valueB :
         *
dtype0*
_output_shapes
: 
├
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/f_accStackV2Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/Const*
_output_shapes
:*
	elem_type0*M
_classC
A?loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh*

stack_name 
и
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/EnterEnterTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/f_acc*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context*
T0*
is_constant(
м
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPushV2StackPushV2Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/Enter:encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh^gradients/Add_1*
T0*'
_output_shapes
:         *
swap_memory( 
ћ
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPopV2
StackPopV2_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub_1*'
_output_shapes
:         *
	elem_type0
╠
_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPopV2/EnterEnterTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
К
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/SumSumNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
┌
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ReshapeReshapeNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Sumkgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
═
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1Mul[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPopV2egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
ш
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/ConstConst*
_output_shapes
: *R
_classH
FDloc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1*
valueB :
         *
dtype0
╠
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/f_accStackV2Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/Const*R
_classH
FDloc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:*
	elem_type0
╗
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/EnterEnterVgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/f_acc*
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
█
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPushV2StackPushV2Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/Enter?encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1^gradients/Add_1*
T0*'
_output_shapes
:         *
swap_memory( 
ў
[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPopV2
StackPopV2agradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*'
_output_shapes
:         *
	elem_type0
л
agradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPopV2/EnterEnterVgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
═
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Sum_1SumPgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Я
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape_1ReshapePgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Sum_1mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*'
_output_shapes
:         *
T0*
Tshape0
Ј
[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/tuple/group_depsNoOpS^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/ReshapeU^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape_1
║
cgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/tuple/control_dependencyIdentityRgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape\^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape*'
_output_shapes
:         
└
egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/tuple/control_dependency_1IdentityTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape_1\^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Reshape_1*'
_output_shapes
:         
Ш
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradQgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:
ћ
]gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/group_depsNoOpY^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGradR^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat
╝
egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityQgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat^^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:         *
T0*d
_classZ
XVloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat
┐
ggradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityXgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGrad^^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
О
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPopV2agradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
П
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGrad[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPopV2cgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/tuple/control_dependency*'
_output_shapes
:         *
T0
м
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_grad/TanhGradTanhGradYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPopV2egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
ш
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMulMatMulegradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/control_dependencyXgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul/Enter*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b(
г
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul/EnterEnter7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read*
_output_shapes

:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
ш
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1MatMul_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPopV2egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
Ш
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/ConstConst*O
_classE
CAloc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat*
valueB :
         *
dtype0*
_output_shapes
: 
Л
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/Const*O
_classE
CAloc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat*

stack_name *
_output_shapes
:*
	elem_type0
├
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/EnterEnterZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
я
`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/Enter<decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat^gradients/Add*
T0*'
_output_shapes
:         *
swap_memory( 
ъ
_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
п
egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
љ
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/group_depsNoOpS^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMulU^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1
╝
dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityRgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul]^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/group_deps*e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul*'
_output_shapes
:         *
T0
╣
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1]^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1*
_output_shapes

:
Ц
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB*    *
dtype0*
_output_shapes
:
╦
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterXgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
Л
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_1`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/NextIteration*
N*
_output_shapes

:: *
T0
 
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*
T0* 
_output_shapes
::
╚
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/AddAdd[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Switch:1ggradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:*
T0
Ь
`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationVgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes
:*
T0
Р
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes
:*
T0
╦
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/ShapeShape=encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:2*
T0*
out_type0*
_output_shapes
:
Ц
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1Const^gradients/Sub_1*
valueB *
dtype0*
_output_shapes
: 
ы
^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsigradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1*2
_output_shapes 
:         :         *
T0
њ
dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/ConstConst*a
_classW
USloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
э
dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/f_accStackV2dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*a
_classW
USloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape
О
dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/EnterEnterdgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
щ
jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/EnterNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape^gradients/Add_1*
_output_shapes
:*
swap_memory( *
T0
Д
igradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ogradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:
В
ogradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnterdgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
═
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/SumSumXgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGrad^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
н
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/ReshapeReshapeLgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Sumigradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
Л
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Sum_1SumXgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGrad`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
«
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Reshape_1ReshapeNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Sum_1Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
Ѕ
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/tuple/group_depsNoOpQ^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/ReshapeS^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Reshape_1
▓
agradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/tuple/control_dependencyIdentityPgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/ReshapeZ^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/tuple/group_deps*'
_output_shapes
:         *
T0*c
_classY
WUloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Reshape
Д
cgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/tuple/control_dependency_1IdentityRgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Reshape_1Z^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Reshape_1*
_output_shapes
: 
р
9gradients/encoder/rnn/while/Switch_5_grad_1/NextIterationNextIterationcgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
Б
Qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ConstConst^gradients/Sub*
dtype0*
_output_shapes
: *
value	B :
б
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/RankConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
А
Ogradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/modFloorModQgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ConstPgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
└
Qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeShape/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul*
T0*
out_type0*
_output_shapes
:
Я
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeNShapeN]gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPopV2_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPopV2_1*
T0*
out_type0*
N* 
_output_shapes
::
у
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/ConstConst*B
_class8
64loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul*
valueB :
         *
dtype0*
_output_shapes
: 
└
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/f_accStackV2Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/Const*B
_class8
64loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul*

stack_name *
_output_shapes
:*
	elem_type0
┐
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/EnterEnterXgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
═
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/Enter/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul^gradients/Add*
T0*'
_output_shapes
:         *
swap_memory( 
џ
]gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2cgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
н
cgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterXgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
о
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/Const_1Const*/
_class%
#!loc:@decoder/rnn/while/Identity_6*
valueB :
         *
dtype0*
_output_shapes
: 
▒
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/f_acc_1StackV2Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/Const_1*
_output_shapes
:*
	elem_type0*/
_class%
#!loc:@decoder/rnn/while/Identity_6*

stack_name 
├
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/Enter_1EnterZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
Й
`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPushV2_1StackPushV2Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/Enter_1decoder/rnn/while/Identity_6^gradients/Add*'
_output_shapes
:         *
swap_memory( *
T0
ъ
_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPopV2_1
StackPopV2egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPopV2_1/Enter^gradients/Sub*
	elem_type0*'
_output_shapes
:         
п
egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPopV2_1/EnterEnterZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/f_acc_1*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0
ј
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ConcatOffsetConcatOffsetOgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/modRgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeNTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
Г
Qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/SliceSlicedgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/control_dependencyXgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ConcatOffsetRgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN*
Index0*
T0*'
_output_shapes
:         
│
Sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Slice_1Slicedgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/control_dependencyZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ConcatOffset:1Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*'
_output_shapes
:         
ј
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/group_depsNoOpR^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/SliceT^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Slice_1
║
dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/control_dependencyIdentityQgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Slice]^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Slice*'
_output_shapes
:         
└
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/control_dependency_1IdentitySgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Slice_1]^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/group_deps*'
_output_shapes
:         *
T0*f
_class\
ZXloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Slice_1
г
Wgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_accConst*
_output_shapes

:*
valueB*    *
dtype0
═
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_acc_1EnterWgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_acc*
is_constant( *
parallel_iterations *
_output_shapes

:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0
м
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_acc_2MergeYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_acc_1_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/NextIteration*
N* 
_output_shapes
:: *
T0
Ё
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/SwitchSwitchYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*(
_output_shapes
::*
T0
╔
Ugradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/AddAddZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/Switch:1fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:*
T0
­
_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/NextIterationNextIterationUgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/Add*
T0*
_output_shapes

:
С
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_acc_3ExitXgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/Switch*
_output_shapes

:*
T0
В
Qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_grad/concatConcatV2Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1_grad/SigmoidGradRgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_grad/TanhGradagradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/tuple/control_dependencyZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2_grad/SigmoidGradWgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/Const*
N*'
_output_shapes
:         *

Tidx0*
T0
Ф
Wgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/ConstConst^gradients/Sub_1*
dtype0*
_output_shapes
: *
value	B :
│
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/ShapeShape/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div*
T0*
out_type0*
_output_shapes
:
Д
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Shape_1Const^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
М
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/StackPopV2Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
■
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/ConstConst*
_output_shapes
: *W
_classM
KIloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Shape*
valueB :
         *
dtype0
┘
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/f_accStackV2Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/Const*
_output_shapes
:*
	elem_type0*W
_classM
KIloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Shape*

stack_name 
├
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/EnterEnterZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
┘
`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/EnterDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Shape^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
Љ
_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
п
egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
░
Bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/MulMuldgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/control_dependencyMgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/StackPopV2*
T0*'
_output_shapes
:         
┘
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/ConstConst*D
_class:
86loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor*
valueB :
         *
dtype0*
_output_shapes
: 
б
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/f_accStackV2Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/Const*D
_class:
86loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor*

stack_name *
_output_shapes
:*
	elem_type0
Ъ
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/EnterEnterHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
д
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/StackPushV2StackPushV2Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/Enter1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor^gradients/Add*
_output_shapes

:*
swap_memory( *
T0
ы
Mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/StackPopV2
StackPopV2Sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/StackPopV2/Enter^gradients/Sub*
_output_shapes

:*
	elem_type0
┤
Sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/StackPopV2/EnterEnterHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
Б
Bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/SumSumBgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/MulTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Х
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/ReshapeReshapeBgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Sum_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
┤
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1MulOgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/StackPopV2dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/control_dependency*'
_output_shapes
:         *
T0
┘
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/ConstConst*B
_class8
64loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div*
valueB :
         *
dtype0*
_output_shapes
: 
ц
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/f_accStackV2Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/Const*B
_class8
64loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div*

stack_name *
_output_shapes
:*
	elem_type0
Б
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/EnterEnterJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
▒
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/StackPushV2StackPushV2Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/Enter/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div^gradients/Add*
T0*'
_output_shapes
:         *
swap_memory( 
■
Ogradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/StackPopV2
StackPopV2Ugradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
И
Ugradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/StackPopV2/EnterEnterJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/f_acc*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0*
is_constant(
Е
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Sum_1SumDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ў
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Reshape_1ReshapeDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Sum_1Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
в
Ogradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/tuple/group_depsNoOpG^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/ReshapeI^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Reshape_1
і
Wgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/tuple/control_dependencyIdentityFgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/ReshapeP^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/tuple/group_deps*Y
_classO
MKloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Reshape*'
_output_shapes
:         *
T0
Є
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/tuple/control_dependency_1IdentityHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Reshape_1P^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/tuple/group_deps*
_output_shapes

:*
T0*[
_classQ
OMloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Reshape_1
Ш
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradQgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:
ћ
]gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/group_depsNoOpY^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGradR^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat
╝
egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityQgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat^^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat*'
_output_shapes
:         
┐
ggradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityXgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGrad^^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
х
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/ShapeShape1decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1*
T0*
out_type0*
_output_shapes
:
Ў
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Shape_1Const^gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
М
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/StackPopV2Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Shape_1*2
_output_shapes 
:         :         *
T0
■
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/ConstConst*W
_classM
KIloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
┘
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/f_accStackV2Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/Const*W
_classM
KIloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
├
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/EnterEnterZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context*
T0*
is_constant(
┘
`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/EnterDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Shape^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
Љ
_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
п
egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
ф
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/RealDivRealDivWgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/tuple/control_dependencyLgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/RealDiv/Const*
T0*'
_output_shapes
:         
А
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/RealDiv/ConstConst^gradients/Sub*
dtype0*
_output_shapes
: *
valueB
 *33s?
Д
Bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/SumSumFgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/RealDivTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Х
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/ReshapeReshapeBgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Sum_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:         *
T0*
Tshape0
╩
Bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/NegNegMgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/StackPopV2*
T0*'
_output_shapes
:         
┘
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/ConstConst*D
_class:
86loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1*
valueB :
         *
dtype0*
_output_shapes
: 
б
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/f_accStackV2Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/Const*D
_class:
86loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1*

stack_name *
_output_shapes
:*
	elem_type0
Ъ
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/EnterEnterHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
»
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/StackPushV2StackPushV2Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/Enter1decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1^gradients/Add*
T0*'
_output_shapes
:         *
swap_memory( 
Щ
Mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/StackPopV2
StackPopV2Sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
┤
Sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/StackPopV2/EnterEnterHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/f_acc*
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
Ќ
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/RealDiv_1RealDivBgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/NegLgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/RealDiv/Const*
T0*'
_output_shapes
:         
Ю
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/RealDiv_2RealDivHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/RealDiv_1Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/RealDiv/Const*
T0*'
_output_shapes
:         
ъ
Bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/mulMulWgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/tuple/control_dependencyHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/RealDiv_2*'
_output_shapes
:         *
T0
Д
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Sum_1SumBgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/mulVgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
љ
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Reshape_1ReshapeDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Sum_1Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
в
Ogradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/tuple/group_depsNoOpG^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/ReshapeI^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Reshape_1
і
Wgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/tuple/control_dependencyIdentityFgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/ReshapeP^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Reshape*'
_output_shapes
:         
 
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/tuple/control_dependency_1IdentityHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Reshape_1P^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Reshape_1*
_output_shapes
: 
С
9gradients/decoder/rnn/while/Switch_6_grad_1/NextIterationNextIterationfgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
ш
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMulMatMulegradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/control_dependencyXgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul/Enter*'
_output_shapes
:         *
transpose_a( *
transpose_b(*
T0
г
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul/EnterEnter7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:*9

frame_name+)gradients/encoder/rnn/while/while_context
ш
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1MatMul_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPopV2egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
Ш
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/ConstConst*O
_classE
CAloc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat*
valueB :
         *
dtype0*
_output_shapes
: 
Л
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/Const*
_output_shapes
:*
	elem_type0*O
_classE
CAloc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat*

stack_name 
├
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/EnterEnterZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context*
T0*
is_constant(
Я
`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/Enter<encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat^gradients/Add_1*'
_output_shapes
:         *
swap_memory( *
T0
а
_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub_1*'
_output_shapes
:         *
	elem_type0
п
egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/f_acc*
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
љ
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/group_depsNoOpS^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMulU^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1
╝
dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityRgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul]^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul*'
_output_shapes
:         
╣
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1]^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*g
_class]
[Yloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1
Ц
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB*    *
dtype0*
_output_shapes
:
╦
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterXgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc*
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0*
is_constant( *
parallel_iterations 
Л
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_1`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/NextIteration*
_output_shapes

:: *
T0*
N
 
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_6* 
_output_shapes
::*
T0
╚
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/AddAdd[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Switch:1ggradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:
Ь
`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationVgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes
:
Р
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes
:*
T0
и
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/ShapeShape1decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1*
_output_shapes
:*
T0*
out_type0
Е
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Shape_1Const^gradients/Sub*
dtype0*
_output_shapes
:*
valueB"      
┘
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsagradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/StackPopV2Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
ѓ
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/ConstConst*
_output_shapes
: *Y
_classO
MKloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Shape*
valueB :
         *
dtype0
▀
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/f_accStackV2\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*Y
_classO
MKloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Shape
К
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/EnterEnter\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
▀
bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/EnterFgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Shape^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
Ћ
agradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ggradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:
▄
ggradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
Д
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/MulMulWgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/tuple/control_dependencyOgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/StackPopV2*
T0*'
_output_shapes
:         
П
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/ConstConst*F
_class<
:8loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/Floor_1*
valueB :
         *
dtype0*
_output_shapes
: 
е
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/f_accStackV2Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/Const*F
_class<
:8loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/Floor_1*

stack_name *
_output_shapes
:*
	elem_type0
Б
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/EnterEnterJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
г
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/StackPushV2StackPushV2Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/Enter3decoder/rnn/while/rnn/multi_rnn_cell/cell_0/Floor_1^gradients/Add*
T0*
_output_shapes

:*
swap_memory( 
ш
Ogradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/StackPopV2
StackPopV2Ugradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub*
_output_shapes

:*
	elem_type0
И
Ugradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/StackPopV2/EnterEnterJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/f_acc*
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
Е
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/SumSumDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/MulVgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╝
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/ReshapeReshapeDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Sumagradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
Ф
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1MulQgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/StackPopV2Wgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
П
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/ConstConst*D
_class:
86loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1*
valueB :
         *
dtype0*
_output_shapes
: 
ф
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/f_accStackV2Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/Const*D
_class:
86loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1*

stack_name *
_output_shapes
:*
	elem_type0
Д
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/EnterEnterLgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
и
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/StackPushV2StackPushV2Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/Enter1decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1^gradients/Add*
T0*'
_output_shapes
:         *
swap_memory( 
ѓ
Qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/StackPopV2
StackPopV2Wgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
╝
Wgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/StackPopV2/EnterEnterLgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
»
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Sum_1SumFgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ъ
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Reshape_1ReshapeFgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Sum_1Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
ы
Qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/tuple/group_depsNoOpI^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/ReshapeK^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Reshape_1
њ
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/tuple/control_dependencyIdentityHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/ReshapeR^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Reshape*'
_output_shapes
:         
Ј
[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/tuple/control_dependency_1IdentityJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Reshape_1R^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Reshape_1*
_output_shapes

:
Ц
Qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ConstConst^gradients/Sub_1*
_output_shapes
: *
value	B :*
dtype0
ц
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/RankConst^gradients/Sub_1*
dtype0*
_output_shapes
: *
value	B :
А
Ogradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/modFloorModQgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ConstPgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
└
Qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeShape/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul*
_output_shapes
:*
T0*
out_type0
Я
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeNShapeN]gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPopV2_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPopV2_1*
T0*
out_type0*
N* 
_output_shapes
::
у
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/ConstConst*
dtype0*
_output_shapes
: *B
_class8
64loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul*
valueB :
         
└
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/f_accStackV2Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/Const*B
_class8
64loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul*

stack_name *
_output_shapes
:*
	elem_type0
┐
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/EnterEnterXgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
¤
^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/Enter/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul^gradients/Add_1*
T0*'
_output_shapes
:         *
swap_memory( 
ю
]gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2cgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^gradients/Sub_1*'
_output_shapes
:         *
	elem_type0
н
cgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterXgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
о
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/Const_1Const*/
_class%
#!loc:@encoder/rnn/while/Identity_6*
valueB :
         *
dtype0*
_output_shapes
: 
▒
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/f_acc_1StackV2Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/Const_1*
	elem_type0*/
_class%
#!loc:@encoder/rnn/while/Identity_6*

stack_name *
_output_shapes
:
├
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/Enter_1EnterZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/f_acc_1*
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
└
`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPushV2_1StackPushV2Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/Enter_1encoder/rnn/while/Identity_6^gradients/Add_1*'
_output_shapes
:         *
swap_memory( *
T0
а
_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPopV2_1
StackPopV2egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPopV2_1/Enter^gradients/Sub_1*'
_output_shapes
:         *
	elem_type0
п
egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPopV2_1/EnterEnterZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
ј
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ConcatOffsetConcatOffsetOgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/modRgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeNTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
Г
Qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/SliceSlicedgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/control_dependencyXgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ConcatOffsetRgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN*'
_output_shapes
:         *
Index0*
T0
│
Sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Slice_1Slicedgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/control_dependencyZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ConcatOffset:1Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*'
_output_shapes
:         
ј
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/group_depsNoOpR^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/SliceT^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Slice_1
║
dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/control_dependencyIdentityQgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Slice]^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/group_deps*'
_output_shapes
:         *
T0*d
_classZ
XVloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Slice
└
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/control_dependency_1IdentitySgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Slice_1]^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Slice_1*'
_output_shapes
:         
г
Wgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_accConst*
dtype0*
_output_shapes

:*
valueB*    
═
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_acc_1EnterWgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:*9

frame_name+)gradients/encoder/rnn/while/while_context
м
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_acc_2MergeYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_acc_1_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/NextIteration*
N* 
_output_shapes
:: *
T0
Ё
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/SwitchSwitchYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_6*
T0*(
_output_shapes
::
╔
Ugradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/AddAddZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/Switch:1fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0*
_output_shapes

:
­
_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/NextIterationNextIterationUgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/Add*
T0*
_output_shapes

:
С
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_acc_3ExitXgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/Switch*
T0*
_output_shapes

:
┴
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/ShapeShape;decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2*
T0*
out_type0*
_output_shapes
:
Џ
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Shape_1Const^gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
┘
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgsBroadcastGradientArgsagradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/StackPopV2Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
ѓ
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *Y
_classO
MKloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Shape*
valueB :
         
▀
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/f_accStackV2\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/Const*Y
_classO
MKloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
К
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/EnterEnter\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
▀
bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/EnterFgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Shape^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
Ћ
agradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ggradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
▄
ggradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
░
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/RealDivRealDivYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/tuple/control_dependencyNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/RealDiv/Const*
T0*'
_output_shapes
:         
Б
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/RealDiv/ConstConst^gradients/Sub*
dtype0*
_output_shapes
: *
valueB
 *33s?
Г
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/SumSumHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/RealDivVgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╝
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/ReshapeReshapeDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Sumagradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
╠
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/NegNegMgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/StackPopV2*'
_output_shapes
:         *
T0
Ю
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/RealDiv_1RealDivDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/NegNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/RealDiv/Const*
T0*'
_output_shapes
:         
Б
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/RealDiv_2RealDivJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/RealDiv_1Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/RealDiv/Const*
T0*'
_output_shapes
:         
ц
Dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/mulMulYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/tuple/control_dependencyJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/RealDiv_2*
T0*'
_output_shapes
:         
Г
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Sum_1SumDgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/mulXgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ќ
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Reshape_1ReshapeFgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Sum_1Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
ы
Qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/tuple/group_depsNoOpI^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/ReshapeK^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Reshape_1
њ
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/tuple/control_dependencyIdentityHgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/ReshapeR^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Reshape*'
_output_shapes
:         
Є
[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/tuple/control_dependency_1IdentityJgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Reshape_1R^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/tuple/group_deps*]
_classS
QOloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Reshape_1*
_output_shapes
: *
T0
│
Dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/ShapeShape/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div*
T0*
out_type0*
_output_shapes
:
Е
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Shape_1Const^gradients/Sub_1*
valueB"      *
dtype0*
_output_shapes
:
М
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/StackPopV2Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
■
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/ConstConst*
_output_shapes
: *W
_classM
KIloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Shape*
valueB :
         *
dtype0
┘
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/f_accStackV2Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/Const*W
_classM
KIloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
├
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/EnterEnterZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
█
`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/EnterDgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Shape^gradients/Add_1*
_output_shapes
:*
swap_memory( *
T0
Њ
_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
п
egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
░
Bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/MulMuldgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/control_dependencyMgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/StackPopV2*'
_output_shapes
:         *
T0
┘
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/ConstConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor*
valueB :
         
б
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/f_accStackV2Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/Const*D
_class:
86loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor*

stack_name *
_output_shapes
:*
	elem_type0
Ъ
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/EnterEnterHgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
е
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/StackPushV2StackPushV2Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/Enter1encoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor^gradients/Add_1*
T0*
_output_shapes

:*
swap_memory( 
з
Mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/StackPopV2
StackPopV2Sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/StackPopV2/Enter^gradients/Sub_1*
_output_shapes

:*
	elem_type0
┤
Sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/StackPopV2/EnterEnterHgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0
Б
Bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/SumSumBgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/MulTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Х
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/ReshapeReshapeBgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Sum_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
┤
Dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1MulOgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/StackPopV2dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
┘
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/ConstConst*B
_class8
64loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div*
valueB :
         *
dtype0*
_output_shapes
: 
ц
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/f_accStackV2Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/Const*
_output_shapes
:*
	elem_type0*B
_class8
64loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div*

stack_name 
Б
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/EnterEnterJgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
│
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/StackPushV2StackPushV2Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/Enter/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div^gradients/Add_1*
T0*'
_output_shapes
:         *
swap_memory( 
ђ
Ogradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/StackPopV2
StackPopV2Ugradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*'
_output_shapes
:         *
	elem_type0
И
Ugradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/StackPopV2/EnterEnterJgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
Е
Dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Sum_1SumDgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ў
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Reshape_1ReshapeDgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Sum_1Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Shape_1*
_output_shapes

:*
T0*
Tshape0
в
Ogradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/tuple/group_depsNoOpG^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/ReshapeI^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Reshape_1
і
Wgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/tuple/control_dependencyIdentityFgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/ReshapeP^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Reshape*'
_output_shapes
:         
Є
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/tuple/control_dependency_1IdentityHgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Reshape_1P^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Reshape_1*
_output_shapes

:
Р
gradients/AddN_3AddNWgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/tuple/control_dependencyYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/tuple/control_dependency*Y
_classO
MKloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Reshape*
N*'
_output_shapes
:         *
T0
¤
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ShapeShape?decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2*
T0*
out_type0*
_output_shapes
:
╬
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1Shape<decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
_output_shapes
:*
T0*
out_type0
њ
`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgskgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:         :         *
T0
ќ
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/ConstConst*c
_classY
WUloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
§
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_accStackV2fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const*
_output_shapes
:*
	elem_type0*c
_classY
WUloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape*

stack_name 
█
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterEnterfgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
§
lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterPgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
Е
kgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
­
qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterfgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
џ
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1*
valueB :
         
Ѓ
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1*e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
▀
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Enterhgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
Ѓ
ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
Г
mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
З
sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterhgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
З
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/MulMulgradients/AddN_3Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPopV2*
T0*'
_output_shapes
:         
­
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/ConstConst*O
_classE
CAloc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
valueB :
         *
dtype0*
_output_shapes
: 
┼
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/f_accStackV2Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/Const*
_output_shapes
:*
	elem_type0*O
_classE
CAloc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*

stack_name 
и
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/EnterEnterTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
м
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPushV2StackPushV2Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/Enter<decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1^gradients/Add*'
_output_shapes
:         *
swap_memory( *
T0
њ
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPopV2
StackPopV2_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
╠
_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPopV2/EnterEnterTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
К
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/SumSumNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
┌
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ReshapeReshapeNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Sumkgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
Э
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1Mul[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPopV2gradients/AddN_3*
T0*'
_output_shapes
:         
ш
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/ConstConst*R
_classH
FDloc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2*
valueB :
         *
dtype0*
_output_shapes
: 
╠
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/f_accStackV2Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/Const*R
_classH
FDloc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2*

stack_name *
_output_shapes
:*
	elem_type0
╗
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/EnterEnterVgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/f_acc*
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
┘
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPushV2StackPushV2Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/Enter?decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2^gradients/Add*
T0*'
_output_shapes
:         *
swap_memory( 
ќ
[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPopV2
StackPopV2agradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
л
agradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPopV2/EnterEnterVgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
═
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Sum_1SumPgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Я
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape_1ReshapePgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Sum_1mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*'
_output_shapes
:         
Ј
[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/tuple/group_depsNoOpS^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ReshapeU^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape_1
║
cgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/tuple/control_dependencyIdentityRgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape\^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape*'
_output_shapes
:         
└
egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/tuple/control_dependency_1IdentityTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape_1\^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape_1*'
_output_shapes
:         
х
Dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/ShapeShape1encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1*
T0*
out_type0*
_output_shapes
:
Џ
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Shape_1Const^gradients/Sub_1*
valueB *
dtype0*
_output_shapes
: 
М
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/StackPopV2Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Shape_1*2
_output_shapes 
:         :         *
T0
■
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/ConstConst*W
_classM
KIloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
┘
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/f_accStackV2Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/Const*W
_classM
KIloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
├
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/EnterEnterZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
█
`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/EnterDgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Shape^gradients/Add_1*
T0*
_output_shapes
:*
swap_memory( 
Њ
_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
п
egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0*
is_constant(
ф
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/RealDivRealDivWgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/tuple/control_dependencyLgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/RealDiv/Const*'
_output_shapes
:         *
T0
Б
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/RealDiv/ConstConst^gradients/Sub_1*
dtype0*
_output_shapes
: *
valueB
 *33s?
Д
Bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/SumSumFgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/RealDivTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Х
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/ReshapeReshapeBgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Sum_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:         *
T0*
Tshape0
╩
Bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/NegNegMgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/StackPopV2*'
_output_shapes
:         *
T0
┘
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/ConstConst*D
_class:
86loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1*
valueB :
         *
dtype0*
_output_shapes
: 
б
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/f_accStackV2Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/Const*D
_class:
86loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1*

stack_name *
_output_shapes
:*
	elem_type0
Ъ
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/EnterEnterHgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/f_acc*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context*
T0*
is_constant(
▒
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/StackPushV2StackPushV2Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/Enter1encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1^gradients/Add_1*
T0*'
_output_shapes
:         *
swap_memory( 
Ч
Mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/StackPopV2
StackPopV2Sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*'
_output_shapes
:         
┤
Sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/StackPopV2/EnterEnterHgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
Ќ
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/RealDiv_1RealDivBgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/NegLgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/RealDiv/Const*
T0*'
_output_shapes
:         
Ю
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/RealDiv_2RealDivHgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/RealDiv_1Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/RealDiv/Const*
T0*'
_output_shapes
:         
ъ
Bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/mulMulWgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/tuple/control_dependencyHgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/RealDiv_2*
T0*'
_output_shapes
:         
Д
Dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Sum_1SumBgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/mulVgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
љ
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Reshape_1ReshapeDgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Sum_1Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
в
Ogradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/tuple/group_depsNoOpG^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/ReshapeI^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Reshape_1
і
Wgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/tuple/control_dependencyIdentityFgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/ReshapeP^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/tuple/group_deps*'
_output_shapes
:         *
T0*Y
_classO
MKloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Reshape
 
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/tuple/control_dependency_1IdentityHgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Reshape_1P^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/tuple/group_deps*
_output_shapes
: *
T0*[
_classQ
OMloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Reshape_1
С
9gradients/encoder/rnn/while/Switch_6_grad_1/NextIterationNextIterationfgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/control_dependency_1*'
_output_shapes
:         *
T0
П
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPopV2cgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
н
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1_grad/TanhGradTanhGradYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPopV2egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/tuple/control_dependency_1*'
_output_shapes
:         *
T0
и
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/ShapeShape1encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1*
T0*
out_type0*
_output_shapes
:
Ф
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Shape_1Const^gradients/Sub_1*
dtype0*
_output_shapes
:*
valueB"      
┘
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsagradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/StackPopV2Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ѓ
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/ConstConst*Y
_classO
MKloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
▀
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/f_accStackV2\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/Const*
	elem_type0*Y
_classO
MKloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Shape*

stack_name *
_output_shapes
:
К
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/EnterEnter\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
р
bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/EnterFgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Shape^gradients/Add_1*
T0*
_output_shapes
:*
swap_memory( 
Ќ
agradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ggradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
▄
ggradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
Д
Dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/MulMulWgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/tuple/control_dependencyOgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/StackPopV2*'
_output_shapes
:         *
T0
П
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/ConstConst*F
_class<
:8loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/Floor_1*
valueB :
         *
dtype0*
_output_shapes
: 
е
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/f_accStackV2Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/Const*
_output_shapes
:*
	elem_type0*F
_class<
:8loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/Floor_1*

stack_name 
Б
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/EnterEnterJgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
«
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/StackPushV2StackPushV2Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/Enter3encoder/rnn/while/rnn/multi_rnn_cell/cell_0/Floor_1^gradients/Add_1*
_output_shapes

:*
swap_memory( *
T0
э
Ogradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/StackPopV2
StackPopV2Ugradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub_1*
_output_shapes

:*
	elem_type0
И
Ugradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/StackPopV2/EnterEnterJgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0
Е
Dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/SumSumDgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/MulVgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╝
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/ReshapeReshapeDgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Sumagradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:         *
T0*
Tshape0
Ф
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1MulQgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/StackPopV2Wgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/tuple/control_dependency*'
_output_shapes
:         *
T0
П
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1*
valueB :
         
ф
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/f_accStackV2Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/Const*D
_class:
86loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1*

stack_name *
_output_shapes
:*
	elem_type0
Д
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/EnterEnterLgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
╣
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/StackPushV2StackPushV2Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/Enter1encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1^gradients/Add_1*'
_output_shapes
:         *
swap_memory( *
T0
ё
Qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/StackPopV2
StackPopV2Wgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*'
_output_shapes
:         *
	elem_type0
╝
Wgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/StackPopV2/EnterEnterLgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
»
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Sum_1SumFgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ъ
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Reshape_1ReshapeFgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Sum_1Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Shape_1*
_output_shapes

:*
T0*
Tshape0
ы
Qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/tuple/group_depsNoOpI^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/ReshapeK^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Reshape_1
њ
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/tuple/control_dependencyIdentityHgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/ReshapeR^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Reshape*'
_output_shapes
:         
Ј
[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/tuple/control_dependency_1IdentityJgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Reshape_1R^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/tuple/group_deps*
_output_shapes

:*
T0*]
_classS
QOloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Reshape_1
х
gradients/AddN_4AddNCgradients/decoder/rnn/while/Merge_3_grad/tuple/control_dependency_1Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1_grad/TanhGrad*
N*'
_output_shapes
:         *
T0*E
_class;
97loc:@gradients/decoder/rnn/while/Switch_3_grad/b_switch
╔
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ShapeShape9decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul*
T0*
out_type0*
_output_shapes
:
═
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1Shape;decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1*
T0*
out_type0*
_output_shapes
:
њ
`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgskgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
ќ
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/ConstConst*
_output_shapes
: *c
_classY
WUloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape*
valueB :
         *
dtype0
§
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/f_accStackV2fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/Const*c
_classY
WUloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
█
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterEnterfgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
§
lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterPgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
Е
kgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
­
qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterfgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
џ
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1Const*e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
Ѓ
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1*e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
▀
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Enterhgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context*
T0*
is_constant(
Ѓ
ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
Г
mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:
З
sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterhgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
Ѕ
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/SumSumgradients/AddN_4`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
┌
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ReshapeReshapeNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Sumkgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:         *
T0*
Tshape0
Ї
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_4bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Я
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1ReshapePgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Sum_1mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*'
_output_shapes
:         
Ј
[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/group_depsNoOpS^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ReshapeU^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1
║
cgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/control_dependencyIdentityRgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape\^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/group_deps*'
_output_shapes
:         *
T0*e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape
└
egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/control_dependency_1IdentityTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1\^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/group_deps*'
_output_shapes
:         *
T0*g
_class]
[Yloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1
┴
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/ShapeShape;encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2*
T0*
out_type0*
_output_shapes
:
Ю
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Shape_1Const^gradients/Sub_1*
valueB *
dtype0*
_output_shapes
: 
┘
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgsBroadcastGradientArgsagradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/StackPopV2Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
ѓ
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/ConstConst*Y
_classO
MKloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
▀
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/f_accStackV2\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/Const*Y
_classO
MKloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
К
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/EnterEnter\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
р
bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/EnterFgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Shape^gradients/Add_1*
_output_shapes
:*
swap_memory( *
T0
Ќ
agradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ggradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
▄
ggradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
░
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/RealDivRealDivYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/tuple/control_dependencyNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/RealDiv/Const*
T0*'
_output_shapes
:         
Ц
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/RealDiv/ConstConst^gradients/Sub_1*
valueB
 *33s?*
dtype0*
_output_shapes
: 
Г
Dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/SumSumHgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/RealDivVgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╝
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/ReshapeReshapeDgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Sumagradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
╠
Dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/NegNegMgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/StackPopV2*
T0*'
_output_shapes
:         
Ю
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/RealDiv_1RealDivDgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/NegNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/RealDiv/Const*
T0*'
_output_shapes
:         
Б
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/RealDiv_2RealDivJgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/RealDiv_1Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/RealDiv/Const*
T0*'
_output_shapes
:         
ц
Dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/mulMulYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/tuple/control_dependencyJgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/RealDiv_2*'
_output_shapes
:         *
T0
Г
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Sum_1SumDgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/mulXgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ќ
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Reshape_1ReshapeFgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Sum_1Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
ы
Qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/tuple/group_depsNoOpI^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/ReshapeK^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Reshape_1
њ
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/tuple/control_dependencyIdentityHgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/ReshapeR^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Reshape*'
_output_shapes
:         *
T0
Є
[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/tuple/control_dependency_1IdentityJgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Reshape_1R^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/tuple/group_deps*
_output_shapes
: *
T0*]
_classS
QOloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Reshape_1
╦
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/ShapeShape=decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*
T0*
out_type0*
_output_shapes
:
г
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1Shapedecoder/rnn/while/Identity_3*
out_type0*
_output_shapes
:*
T0
ї
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsigradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2kgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
њ
dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/ConstConst*a
_classW
USloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
э
dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStackV2dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/Const*a
_classW
USloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
О
dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/EnterEnterdgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
э
jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/EnterNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
Ц
igradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ogradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
В
ogradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterdgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
ќ
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1Const*c
_classY
WUloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
§
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1StackV2fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1*

stack_name *
_output_shapes
:*
	elem_type0*c
_classY
WUloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1
█
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Enterfgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
§
lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
Е
kgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
­
qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterfgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0*
is_constant(
├
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/MulMulcgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/control_dependencyWgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPopV2*
T0*'
_output_shapes
:         
╬
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/ConstConst*/
_class%
#!loc:@decoder/rnn/while/Identity_3*
valueB :
         *
dtype0*
_output_shapes
: 
А
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/f_accStackV2Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/Const*
	elem_type0*/
_class%
#!loc:@decoder/rnn/while/Identity_3*

stack_name *
_output_shapes
:
│
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/EnterEnterRgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
«
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPushV2StackPushV2Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/Enterdecoder/rnn/while/Identity_3^gradients/Add*
T0*'
_output_shapes
:         *
swap_memory( 
ј
Wgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPopV2
StackPopV2]gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*'
_output_shapes
:         
╚
]gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPopV2/EnterEnterRgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
┴
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/SumSumLgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
н
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/ReshapeReshapeLgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Sumigradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
К
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1MulYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPopV2cgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
ы
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/ConstConst*P
_classF
DBloc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*
valueB :
         *
dtype0*
_output_shapes
: 
к
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/f_accStackV2Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/Const*P
_classF
DBloc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*

stack_name *
_output_shapes
:*
	elem_type0
и
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/EnterEnterTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context*
T0
М
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPushV2StackPushV2Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/Enter=decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid^gradients/Add*
T0*'
_output_shapes
:         *
swap_memory( 
њ
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPopV2
StackPopV2_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
╠
_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPopV2/EnterEnterTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/f_acc*
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
К
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Sum_1SumNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
┌
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape_1ReshapeNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Sum_1kgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*'
_output_shapes
:         *
T0*
Tshape0
Ѕ
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/tuple/group_depsNoOpQ^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/ReshapeS^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape_1
▓
agradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/tuple/control_dependencyIdentityPgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/ReshapeZ^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/tuple/group_deps*c
_classY
WUloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape*'
_output_shapes
:         *
T0
И
cgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/tuple/control_dependency_1IdentityRgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape_1Z^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape_1*'
_output_shapes
:         
¤
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ShapeShape?decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*
T0*
out_type0*
_output_shapes
:
╠
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1Shape:decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh*
T0*
out_type0*
_output_shapes
:
њ
`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgskgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
ќ
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/ConstConst*c
_classY
WUloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
§
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_accStackV2fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*c
_classY
WUloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape
█
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterEnterfgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context*
T0
§
lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterPgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
Е
kgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
­
qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterfgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
џ
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1*
valueB :
         
Ѓ
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1*
_output_shapes
:*
	elem_type0*e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1*

stack_name 
▀
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Enterhgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
Ѓ
ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
Г
mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
З
sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterhgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
╔
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/MulMulegradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/control_dependency_1Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPopV2*
T0*'
_output_shapes
:         
Ь
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/ConstConst*M
_classC
A?loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh*
valueB :
         *
dtype0*
_output_shapes
: 
├
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/f_accStackV2Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/Const*M
_classC
A?loc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh*

stack_name *
_output_shapes
:*
	elem_type0
и
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/EnterEnterTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/f_acc*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context*
T0*
is_constant(
л
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPushV2StackPushV2Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/Enter:decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh^gradients/Add*'
_output_shapes
:         *
swap_memory( *
T0
њ
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPopV2
StackPopV2_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
╠
_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPopV2/EnterEnterTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
К
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/SumSumNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
┌
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ReshapeReshapeNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Sumkgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2*
Tshape0*'
_output_shapes
:         *
T0
═
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1Mul[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPopV2egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/control_dependency_1*'
_output_shapes
:         *
T0
ш
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/ConstConst*
_output_shapes
: *R
_classH
FDloc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*
valueB :
         *
dtype0
╠
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/f_accStackV2Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/Const*R
_classH
FDloc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:*
	elem_type0
╗
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/EnterEnterVgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
┘
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPushV2StackPushV2Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/Enter?decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1^gradients/Add*
T0*'
_output_shapes
:         *
swap_memory( 
ќ
[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPopV2
StackPopV2agradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
л
agradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPopV2/EnterEnterVgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
═
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Sum_1SumPgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Я
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape_1ReshapePgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Sum_1mgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*'
_output_shapes
:         *
T0*
Tshape0
Ј
[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/tuple/group_depsNoOpS^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ReshapeU^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape_1
║
cgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/tuple/control_dependencyIdentityRgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape\^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape*'
_output_shapes
:         
└
egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/tuple/control_dependency_1IdentityTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape_1\^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/tuple/group_deps*'
_output_shapes
:         *
T0*g
_class]
[Yloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape_1
Р
gradients/AddN_5AddNWgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/tuple/control_dependencyYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/tuple/control_dependency*'
_output_shapes
:         *
T0*Y
_classO
MKloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Reshape*
N
¤
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ShapeShape?encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2*
T0*
out_type0*
_output_shapes
:
╬
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1Shape<encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
T0*
out_type0*
_output_shapes
:
њ
`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgskgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:         :         *
T0
ќ
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/ConstConst*c
_classY
WUloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
§
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_accStackV2fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*c
_classY
WUloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape
█
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterEnterfgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
 
lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterPgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape^gradients/Add_1*
T0*
_output_shapes
:*
swap_memory( 
Ф
kgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
­
qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterfgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
џ
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1Const*e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
Ѓ
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1*e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
▀
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Enterhgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
Ё
ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1^gradients/Add_1*
_output_shapes
:*
swap_memory( *
T0
»
mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
З
sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterhgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0*
is_constant(
З
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/MulMulgradients/AddN_5Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPopV2*
T0*'
_output_shapes
:         
­
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/ConstConst*O
_classE
CAloc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
valueB :
         *
dtype0*
_output_shapes
: 
┼
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/f_accStackV2Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/Const*O
_classE
CAloc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*

stack_name *
_output_shapes
:*
	elem_type0
и
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/EnterEnterTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
н
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPushV2StackPushV2Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/Enter<encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1^gradients/Add_1*'
_output_shapes
:         *
swap_memory( *
T0
ћ
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPopV2
StackPopV2_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub_1*'
_output_shapes
:         *
	elem_type0
╠
_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPopV2/EnterEnterTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/f_acc*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0*
is_constant(
К
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/SumSumNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
┌
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ReshapeReshapeNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Sumkgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
Э
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1Mul[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPopV2gradients/AddN_5*'
_output_shapes
:         *
T0
ш
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/ConstConst*R
_classH
FDloc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2*
valueB :
         *
dtype0*
_output_shapes
: 
╠
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/f_accStackV2Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/Const*
_output_shapes
:*
	elem_type0*R
_classH
FDloc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2*

stack_name 
╗
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/EnterEnterVgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
█
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPushV2StackPushV2Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/Enter?encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2^gradients/Add_1*
T0*'
_output_shapes
:         *
swap_memory( 
ў
[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPopV2
StackPopV2agradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*'
_output_shapes
:         *
	elem_type0
л
agradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPopV2/EnterEnterVgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
═
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Sum_1SumPgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Я
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape_1ReshapePgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Sum_1mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*'
_output_shapes
:         
Ј
[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/tuple/group_depsNoOpS^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/ReshapeU^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape_1
║
cgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/tuple/control_dependencyIdentityRgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape\^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape*'
_output_shapes
:         
└
egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/tuple/control_dependency_1IdentityTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape_1\^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/tuple/group_deps*'
_output_shapes
:         *
T0*g
_class]
[Yloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Reshape_1
О
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPopV2agradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
П
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGrad[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPopV2cgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/tuple/control_dependency*'
_output_shapes
:         *
T0
м
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_grad/TanhGradTanhGradYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPopV2egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
П
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPopV2cgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
н
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1_grad/TanhGradTanhGradYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPopV2egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
╦
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/ShapeShape=decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:2*
_output_shapes
:*
T0*
out_type0
Б
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1Const^gradients/Sub*
dtype0*
_output_shapes
: *
valueB 
ы
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsigradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
њ
dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/ConstConst*a
_classW
USloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
э
dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/f_accStackV2dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/Const*a
_classW
USloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
О
dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/EnterEnterdgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
э
jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/EnterNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
Ц
igradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ogradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
В
ogradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnterdgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
═
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/SumSumXgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGrad^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
н
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/ReshapeReshapeLgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Sumigradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
Л
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Sum_1SumXgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGrad`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
«
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Reshape_1ReshapeNgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Sum_1Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ѕ
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/tuple/group_depsNoOpQ^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/ReshapeS^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Reshape_1
▓
agradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/tuple/control_dependencyIdentityPgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/ReshapeZ^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Reshape*'
_output_shapes
:         
Д
cgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/tuple/control_dependency_1IdentityRgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Reshape_1Z^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Reshape_1*
_output_shapes
: 
р
9gradients/decoder/rnn/while/Switch_3_grad_1/NextIterationNextIterationcgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
х
gradients/AddN_6AddNCgradients/encoder/rnn/while/Merge_3_grad/tuple/control_dependency_1Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1_grad/TanhGrad*E
_class;
97loc:@gradients/encoder/rnn/while/Switch_3_grad/b_switch*
N*'
_output_shapes
:         *
T0
╔
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ShapeShape9encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul*
T0*
out_type0*
_output_shapes
:
═
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1Shape;encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1*
T0*
out_type0*
_output_shapes
:
њ
`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgskgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:         :         *
T0
ќ
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/ConstConst*
_output_shapes
: *c
_classY
WUloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape*
valueB :
         *
dtype0
§
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/f_accStackV2fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/Const*c
_classY
WUloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
█
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterEnterfgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
 
lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterPgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape^gradients/Add_1*
_output_shapes
:*
swap_memory( *
T0
Ф
kgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
­
qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterfgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
џ
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1Const*e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
Ѓ
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1*e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
▀
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Enterhgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
Ё
ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1^gradients/Add_1*
T0*
_output_shapes
:*
swap_memory( 
»
mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
З
sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterhgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
Ѕ
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/SumSumgradients/AddN_6`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
┌
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ReshapeReshapeNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Sumkgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
Ї
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_6bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Я
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1ReshapePgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Sum_1mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*'
_output_shapes
:         
Ј
[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/group_depsNoOpS^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/ReshapeU^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1
║
cgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/control_dependencyIdentityRgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape\^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/group_deps*'
_output_shapes
:         *
T0*e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape
└
egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/control_dependency_1IdentityTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1\^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Reshape_1*'
_output_shapes
:         
В
Qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_grad/concatConcatV2Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1_grad/SigmoidGradRgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_grad/TanhGradagradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/tuple/control_dependencyZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2_grad/SigmoidGradWgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/Const*
T0*
N*'
_output_shapes
:         @*

Tidx0
Е
Wgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/ConstConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
╦
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/ShapeShape=encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*
T0*
out_type0*
_output_shapes
:
г
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1Shapeencoder/rnn/while/Identity_3*
_output_shapes
:*
T0*
out_type0
ї
^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsigradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2kgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:         :         *
T0
њ
dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/ConstConst*a
_classW
USloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
э
dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStackV2dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/Const*a
_classW
USloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
О
dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/EnterEnterdgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
щ
jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/EnterNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape^gradients/Add_1*
T0*
_output_shapes
:*
swap_memory( 
Д
igradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ogradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
В
ogradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterdgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
ќ
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1Const*c
_classY
WUloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
§
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1StackV2fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1*c
_classY
WUloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
█
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Enterfgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
 
lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1^gradients/Add_1*
_output_shapes
:*
swap_memory( *
T0
Ф
kgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
­
qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterfgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0
├
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/MulMulcgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/control_dependencyWgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPopV2*
T0*'
_output_shapes
:         
╬
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/ConstConst*/
_class%
#!loc:@encoder/rnn/while/Identity_3*
valueB :
         *
dtype0*
_output_shapes
: 
А
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/f_accStackV2Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/Const*/
_class%
#!loc:@encoder/rnn/while/Identity_3*

stack_name *
_output_shapes
:*
	elem_type0
│
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/EnterEnterRgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
░
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPushV2StackPushV2Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/Enterencoder/rnn/while/Identity_3^gradients/Add_1*'
_output_shapes
:         *
swap_memory( *
T0
љ
Wgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPopV2
StackPopV2]gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPopV2/Enter^gradients/Sub_1*'
_output_shapes
:         *
	elem_type0
╚
]gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPopV2/EnterEnterRgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
┴
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/SumSumLgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
н
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/ReshapeReshapeLgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Sumigradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:         *
T0*
Tshape0
К
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1MulYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPopV2cgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
ы
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/ConstConst*P
_classF
DBloc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*
valueB :
         *
dtype0*
_output_shapes
: 
к
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/f_accStackV2Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/Const*P
_classF
DBloc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid*

stack_name *
_output_shapes
:*
	elem_type0
и
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/EnterEnterTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
Н
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPushV2StackPushV2Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/Enter=encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid^gradients/Add_1*
T0*'
_output_shapes
:         *
swap_memory( 
ћ
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPopV2
StackPopV2_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*'
_output_shapes
:         *
	elem_type0
╠
_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPopV2/EnterEnterTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
К
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Sum_1SumNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
┌
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape_1ReshapeNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Sum_1kgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*'
_output_shapes
:         
Ѕ
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/tuple/group_depsNoOpQ^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/ReshapeS^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape_1
▓
agradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/tuple/control_dependencyIdentityPgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/ReshapeZ^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape*'
_output_shapes
:         
И
cgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/tuple/control_dependency_1IdentityRgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape_1Z^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/tuple/group_deps*e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Reshape_1*'
_output_shapes
:         *
T0
¤
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ShapeShape?encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*
T0*
out_type0*
_output_shapes
:
╠
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1Shape:encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh*
_output_shapes
:*
T0*
out_type0
њ
`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgskgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
ќ
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *c
_classY
WUloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape*
valueB :
         
§
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_accStackV2fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const*c
_classY
WUloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
█
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterEnterfgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context*
T0
 
lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterPgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape^gradients/Add_1*
T0*
_output_shapes
:*
swap_memory( 
Ф
kgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
­
qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterfgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
џ
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1Const*
_output_shapes
: *e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1*
valueB :
         *
dtype0
Ѓ
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1*e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
▀
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Enterhgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
Ё
ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1^gradients/Add_1*
T0*
_output_shapes
:*
swap_memory( 
»
mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
З
sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterhgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0*
is_constant(
╔
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/MulMulegradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/control_dependency_1Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPopV2*'
_output_shapes
:         *
T0
Ь
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/ConstConst*
_output_shapes
: *M
_classC
A?loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh*
valueB :
         *
dtype0
├
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/f_accStackV2Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/Const*M
_classC
A?loc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh*

stack_name *
_output_shapes
:*
	elem_type0
и
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/EnterEnterTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
м
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPushV2StackPushV2Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/Enter:encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh^gradients/Add_1*
T0*'
_output_shapes
:         *
swap_memory( 
ћ
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPopV2
StackPopV2_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*'
_output_shapes
:         
╠
_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPopV2/EnterEnterTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0
К
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/SumSumNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
┌
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ReshapeReshapeNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Sumkgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2*'
_output_shapes
:         *
T0*
Tshape0
═
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1Mul[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPopV2egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
ш
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *R
_classH
FDloc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*
valueB :
         
╠
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/f_accStackV2Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/Const*
_output_shapes
:*
	elem_type0*R
_classH
FDloc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1*

stack_name 
╗
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/EnterEnterVgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context
█
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPushV2StackPushV2Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/Enter?encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1^gradients/Add_1*
T0*'
_output_shapes
:         *
swap_memory( 
ў
[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPopV2
StackPopV2agradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*'
_output_shapes
:         *
	elem_type0
л
agradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPopV2/EnterEnterVgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/f_acc*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0*
is_constant(
═
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Sum_1SumPgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Я
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape_1ReshapePgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Sum_1mgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
Tshape0*'
_output_shapes
:         *
T0
Ј
[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/tuple/group_depsNoOpS^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/ReshapeU^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape_1
║
cgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/tuple/control_dependencyIdentityRgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape\^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/tuple/group_deps*'
_output_shapes
:         *
T0*e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape
└
egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/tuple/control_dependency_1IdentityTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape_1\^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Reshape_1*'
_output_shapes
:         
Ш
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradQgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:@
ћ
]gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/group_depsNoOpY^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGradR^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat
╝
egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityQgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat^^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat*'
_output_shapes
:         @
┐
ggradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityXgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGrad^^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
О
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPopV2agradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/tuple/control_dependency*'
_output_shapes
:         *
T0
П
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGrad[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPopV2cgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/tuple/control_dependency*'
_output_shapes
:         *
T0
м
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_grad/TanhGradTanhGradYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPopV2egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
ш
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMulMatMulegradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/control_dependencyXgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul/Enter*'
_output_shapes
:         *
transpose_a( *
transpose_b(*
T0
г
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul/EnterEnter7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read*
_output_shapes

:@*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
ш
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1MatMul_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPopV2egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
Ш
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/ConstConst*
_output_shapes
: *O
_classE
CAloc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat*
valueB :
         *
dtype0
Л
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/Const*O
_classE
CAloc:@decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat*

stack_name *
_output_shapes
:*
	elem_type0
├
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/EnterEnterZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
я
`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/Enter<decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat^gradients/Add*
T0*'
_output_shapes
:         *
swap_memory( 
ъ
_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
п
egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0
љ
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/group_depsNoOpS^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMulU^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1
╝
dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityRgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul]^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/group_deps*e
_class[
YWloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul*'
_output_shapes
:         *
T0
╣
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1]^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1*
_output_shapes

:@
Ц
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB@*    *
dtype0*
_output_shapes
:@
╦
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterXgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:@*9

frame_name+)gradients/decoder/rnn/while/while_context
Л
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_1`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes

:@: 
 
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2* 
_output_shapes
:@:@*
T0
╚
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/AddAdd[gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Switch:1ggradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:@
Ь
`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationVgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes
:@
Р
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes
:@
╦
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/ShapeShape=encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:2*
_output_shapes
:*
T0*
out_type0
Ц
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1Const^gradients/Sub_1*
valueB *
dtype0*
_output_shapes
: 
ы
^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsigradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
њ
dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/ConstConst*a
_classW
USloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
э
dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/f_accStackV2dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/Const*a
_classW
USloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
О
dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/EnterEnterdgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context*
T0*
is_constant(
щ
jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/EnterNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape^gradients/Add_1*
T0*
_output_shapes
:*
swap_memory( 
Д
igradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ogradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
В
ogradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnterdgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
═
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/SumSumXgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGrad^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
н
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/ReshapeReshapeLgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Sumigradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
Л
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Sum_1SumXgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGrad`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
«
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Reshape_1ReshapeNgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Sum_1Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ѕ
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/tuple/group_depsNoOpQ^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/ReshapeS^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Reshape_1
▓
agradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/tuple/control_dependencyIdentityPgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/ReshapeZ^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/tuple/group_deps*'
_output_shapes
:         *
T0*c
_classY
WUloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Reshape
Д
cgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/tuple/control_dependency_1IdentityRgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Reshape_1Z^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/tuple/group_deps*e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Reshape_1*
_output_shapes
: *
T0
р
9gradients/encoder/rnn/while/Switch_3_grad_1/NextIterationNextIterationcgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
Б
Qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ConstConst^gradients/Sub*
dtype0*
_output_shapes
: *
value	B :
б
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/RankConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
А
Ogradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/modFloorModQgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ConstPgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
┤
Qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeShape#decoder/rnn/while/TensorArrayReadV3*
_output_shapes
:*
T0*
out_type0
Я
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeNShapeN]gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPopV2_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPopV2_1*
T0*
out_type0*
N* 
_output_shapes
::
█
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/ConstConst*
_output_shapes
: *6
_class,
*(loc:@decoder/rnn/while/TensorArrayReadV3*
valueB :
         *
dtype0
┤
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/f_accStackV2Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/Const*6
_class,
*(loc:@decoder/rnn/while/TensorArrayReadV3*

stack_name *
_output_shapes
:*
	elem_type0
┐
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/EnterEnterXgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
┴
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/Enter#decoder/rnn/while/TensorArrayReadV3^gradients/Add*
T0*'
_output_shapes
:         *
swap_memory( 
џ
]gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2cgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
н
cgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterXgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
о
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/Const_1Const*/
_class%
#!loc:@decoder/rnn/while/Identity_4*
valueB :
         *
dtype0*
_output_shapes
: 
▒
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/f_acc_1StackV2Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/Const_1*/
_class%
#!loc:@decoder/rnn/while/Identity_4*

stack_name *
_output_shapes
:*
	elem_type0
├
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/Enter_1EnterZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*/

frame_name!decoder/rnn/while/while_context
Й
`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPushV2_1StackPushV2Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/Enter_1decoder/rnn/while/Identity_4^gradients/Add*
T0*'
_output_shapes
:         *
swap_memory( 
ъ
_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPopV2_1
StackPopV2egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPopV2_1/Enter^gradients/Sub*'
_output_shapes
:         *
	elem_type0
п
egradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPopV2_1/EnterEnterZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/decoder/rnn/while/while_context
ј
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ConcatOffsetConcatOffsetOgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/modRgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeNTgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
Г
Qgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/SliceSlicedgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/control_dependencyXgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ConcatOffsetRgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN*
Index0*
T0*'
_output_shapes
:         
│
Sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Slice_1Slicedgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/control_dependencyZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ConcatOffset:1Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*'
_output_shapes
:         
ј
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/tuple/group_depsNoOpR^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/SliceT^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Slice_1
║
dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/tuple/control_dependencyIdentityQgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Slice]^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/tuple/group_deps*'
_output_shapes
:         *
T0*d
_classZ
XVloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Slice
└
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/tuple/control_dependency_1IdentitySgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Slice_1]^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Slice_1*'
_output_shapes
:         
г
Wgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_accConst*
valueB@*    *
dtype0*
_output_shapes

:@
═
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_acc_1EnterWgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_acc*
parallel_iterations *
_output_shapes

:@*9

frame_name+)gradients/decoder/rnn/while/while_context*
T0*
is_constant( 
м
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_acc_2MergeYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_acc_1_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/NextIteration*
N* 
_output_shapes
:@: *
T0
Ё
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/SwitchSwitchYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*(
_output_shapes
:@:@*
T0
╔
Ugradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/AddAddZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/Switch:1fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0*
_output_shapes

:@
­
_gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/NextIterationNextIterationUgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/Add*
T0*
_output_shapes

:@
С
Ygradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_acc_3ExitXgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/Switch*
T0*
_output_shapes

:@
В
Qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_grad/concatConcatV2Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1_grad/SigmoidGradRgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_grad/TanhGradagradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/tuple/control_dependencyZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2_grad/SigmoidGradWgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/Const*

Tidx0*
T0*
N*'
_output_shapes
:         @
Ф
Wgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/ConstConst^gradients/Sub_1*
value	B :*
dtype0*
_output_shapes
: 
Ш
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradQgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:@
ћ
]gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/group_depsNoOpY^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGradR^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat
╝
egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityQgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat^^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat*'
_output_shapes
:         @
┐
ggradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityXgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGrad^^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
С
9gradients/decoder/rnn/while/Switch_4_grad_1/NextIterationNextIterationfgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
ш
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMulMatMulegradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/control_dependencyXgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul/Enter*'
_output_shapes
:         *
transpose_a( *
transpose_b(*
T0
г
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul/EnterEnter7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read*
is_constant(*
parallel_iterations *
_output_shapes

:@*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0
ш
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1MatMul_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPopV2egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:@*
transpose_a(*
transpose_b( *
T0
Ш
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/ConstConst*O
_classE
CAloc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat*
valueB :
         *
dtype0*
_output_shapes
: 
Л
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/Const*
_output_shapes
:*
	elem_type0*O
_classE
CAloc:@encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat*

stack_name 
├
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/EnterEnterZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/f_acc*
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
Я
`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/Enter<encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat^gradients/Add_1*'
_output_shapes
:         *
swap_memory( *
T0
а
_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub_1*'
_output_shapes
:         *
	elem_type0
п
egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/f_acc*
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
љ
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/group_depsNoOpS^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMulU^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1
╝
dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityRgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul]^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul*'
_output_shapes
:         
╣
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1]^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/group_deps*
_output_shapes

:@*
T0*g
_class]
[Yloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1
Ц
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB@*    *
dtype0*
_output_shapes
:@
╦
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterXgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc*
_output_shapes
:@*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0*
is_constant( *
parallel_iterations 
Л
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_1`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/NextIteration*
N*
_output_shapes

:@: *
T0
 
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_6*
T0* 
_output_shapes
:@:@
╚
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/AddAdd[gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Switch:1ggradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:@
Ь
`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationVgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes
:@
Р
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes
:@
Ц
Qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ConstConst^gradients/Sub_1*
_output_shapes
: *
value	B :*
dtype0
ц
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/RankConst^gradients/Sub_1*
value	B :*
dtype0*
_output_shapes
: 
А
Ogradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/modFloorModQgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ConstPgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
┤
Qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeShape#encoder/rnn/while/TensorArrayReadV3*
_output_shapes
:*
T0*
out_type0
Я
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeNShapeN]gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPopV2_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPopV2_1*
out_type0*
N* 
_output_shapes
::*
T0
█
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/ConstConst*6
_class,
*(loc:@encoder/rnn/while/TensorArrayReadV3*
valueB :
         *
dtype0*
_output_shapes
: 
┤
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/f_accStackV2Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/Const*6
_class,
*(loc:@encoder/rnn/while/TensorArrayReadV3*

stack_name *
_output_shapes
:*
	elem_type0
┐
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/EnterEnterXgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/f_acc*
parallel_iterations *
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context*
T0*
is_constant(
├
^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/Enter#encoder/rnn/while/TensorArrayReadV3^gradients/Add_1*
T0*'
_output_shapes
:         *
swap_memory( 
ю
]gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2cgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*'
_output_shapes
:         
н
cgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterXgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/f_acc*
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
о
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/Const_1Const*/
_class%
#!loc:@encoder/rnn/while/Identity_4*
valueB :
         *
dtype0*
_output_shapes
: 
▒
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/f_acc_1StackV2Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/Const_1*/
_class%
#!loc:@encoder/rnn/while/Identity_4*

stack_name *
_output_shapes
:*
	elem_type0
├
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/Enter_1EnterZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/f_acc_1*
_output_shapes
:*/

frame_name!encoder/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
└
`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPushV2_1StackPushV2Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/Enter_1encoder/rnn/while/Identity_4^gradients/Add_1*'
_output_shapes
:         *
swap_memory( *
T0
а
_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPopV2_1
StackPopV2egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPopV2_1/Enter^gradients/Sub_1*'
_output_shapes
:         *
	elem_type0
п
egradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPopV2_1/EnterEnterZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)gradients/encoder/rnn/while/while_context
ј
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ConcatOffsetConcatOffsetOgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/modRgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeNTgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
Г
Qgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/SliceSlicedgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/control_dependencyXgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ConcatOffsetRgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN*
Index0*
T0*'
_output_shapes
:         
│
Sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Slice_1Slicedgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/control_dependencyZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ConcatOffset:1Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN:1*'
_output_shapes
:         *
Index0*
T0
ј
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/tuple/group_depsNoOpR^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/SliceT^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Slice_1
║
dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/tuple/control_dependencyIdentityQgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Slice]^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Slice*'
_output_shapes
:         
└
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/tuple/control_dependency_1IdentitySgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Slice_1]^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Slice_1*'
_output_shapes
:         
г
Wgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_accConst*
valueB@*    *
dtype0*
_output_shapes

:@
═
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_acc_1EnterWgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:@*9

frame_name+)gradients/encoder/rnn/while/while_context
м
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_acc_2MergeYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_acc_1_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N* 
_output_shapes
:@: 
Ё
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/SwitchSwitchYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_6*(
_output_shapes
:@:@*
T0
╔
Ugradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/AddAddZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/Switch:1fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0*
_output_shapes

:@
­
_gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/NextIterationNextIterationUgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/Add*
_output_shapes

:@*
T0
С
Ygradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_acc_3ExitXgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/Switch*
T0*
_output_shapes

:@
С
9gradients/encoder/rnn/while/Switch_4_grad_1/NextIterationNextIterationfgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
Б
beta1_power/initial_valueConst*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
┤
beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
	container *
shape: 
М
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes
: 
Ј
beta1_power/readIdentitybeta1_power*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
_output_shapes
: 
Б
beta2_power/initial_valueConst*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
valueB
 *wЙ?*
dtype0*
_output_shapes
: 
┤
beta2_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias
М
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Ј
beta2_power/readIdentitybeta2_power*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
_output_shapes
: 
ы
Yencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
valueB"   @   *
dtype0*
_output_shapes
:
█
Oencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
 
Iencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/Initializer/zerosFillYencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorOencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/Initializer/zeros/Const*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*

index_type0*
_output_shapes

:@
Ы
7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
т
>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/AssignAssign7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdamIencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ы
<encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/readIdentity7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
_output_shapes

:@
з
[encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
valueB"   @   
П
Qencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ё
Kencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/Initializer/zerosFill[encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorQencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/Initializer/zeros/Const*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*

index_type0*
_output_shapes

:@
З
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1
VariableV2*
shared_name *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
	container *
shape
:@*
dtype0*
_output_shapes

:@
в
@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/AssignAssign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1Kencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
validate_shape(*
_output_shapes

:@
ш
>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/readIdentity9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
_output_shapes

:@
┘
Gencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam/Initializer/zerosConst*
_output_shapes
:@*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
valueB@*    *
dtype0
Т
5encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
	container *
shape:@
┘
<encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam/AssignAssign5encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdamGencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam/Initializer/zeros*
T0*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
у
:encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam/readIdentity5encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam*
T0*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
_output_shapes
:@
█
Iencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:@*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
valueB@*    
У
7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
	container 
▀
>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1/AssignAssign7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1Iencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias
в
<encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1/readIdentity7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
_output_shapes
:@*
T0
т
Iencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam/Initializer/zerosConst*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
valueB*    *
dtype0*
_output_shapes

:
Ы
7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam
VariableV2*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
т
>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam/AssignAssign7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdamIencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam/Initializer/zeros*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
validate_shape(*
_output_shapes

:
ы
<encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam/readIdentity7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam*
_output_shapes

:*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
у
Kencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1/Initializer/zerosConst*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
valueB*    *
dtype0*
_output_shapes

:
З
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1
VariableV2*
shared_name *E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:
в
@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1/AssignAssign9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1Kencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
validate_shape(*
_output_shapes

:
ш
>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1/readIdentity9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1*
_output_shapes

:*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
┘
Gencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam/Initializer/zerosConst*
_output_shapes
:*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
valueB*    *
dtype0
Т
5encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
	container *
shape:
┘
<encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam/AssignAssign5encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdamGencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
validate_shape(*
_output_shapes
:
у
:encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam/readIdentity5encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam*
T0*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
_output_shapes
:
█
Iencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
valueB*    
У
7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1
VariableV2*
shared_name *C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
	container *
shape:*
dtype0*
_output_shapes
:
▀
>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1/AssignAssign7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1Iencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias
в
<encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1/readIdentity7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1*
T0*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
_output_shapes
:
ы
Ydecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
valueB"   @   *
dtype0*
_output_shapes
:
█
Odecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
 
Idecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/Initializer/zerosFillYdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorOdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/Initializer/zeros/Const*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*

index_type0*
_output_shapes

:@
Ы
7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
	container *
shape
:@
т
>decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/AssignAssign7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdamIdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/Initializer/zeros*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
ы
<decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/readIdentity7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam*
_output_shapes

:@*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
з
[decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
valueB"   @   *
dtype0
П
Qdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ё
Kdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/Initializer/zerosFill[decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorQdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes

:@*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*

index_type0
З
9decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
	container *
shape
:@
в
@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/AssignAssign9decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1Kdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
validate_shape(*
_output_shapes

:@
ш
>decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/readIdentity9decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
_output_shapes

:@
┘
Gdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam/Initializer/zerosConst*
_output_shapes
:@*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
valueB@*    *
dtype0
Т
5decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias
┘
<decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam/AssignAssign5decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdamGdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes
:@
у
:decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam/readIdentity5decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
_output_shapes
:@
█
Idecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:@*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
valueB@*    
У
7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
	container *
shape:@
▀
>decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1/AssignAssign7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1Idecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes
:@
в
<decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1/readIdentity7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1*
_output_shapes
:@*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias
т
Idecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam/Initializer/zerosConst*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
valueB*    *
dtype0*
_output_shapes

:
Ы
7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
	container *
shape
:
т
>decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam/AssignAssign7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdamIdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ы
<decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam/readIdentity7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
_output_shapes

:
у
Kdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1/Initializer/zerosConst*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
valueB*    *
dtype0*
_output_shapes

:
З
9decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
	container *
shape
:
в
@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1/AssignAssign9decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1Kdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
validate_shape(*
_output_shapes

:
ш
>decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1/readIdentity9decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
_output_shapes

:
┘
Gdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam/Initializer/zerosConst*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
valueB*    *
dtype0*
_output_shapes
:
Т
5decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
	container *
shape:
┘
<decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam/AssignAssign5decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdamGdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
validate_shape(*
_output_shapes
:
у
:decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam/readIdentity5decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
_output_shapes
:
█
Idecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1/Initializer/zerosConst*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
valueB*    *
dtype0*
_output_shapes
:
У
7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
	container *
shape:
▀
>decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1/AssignAssign7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1Idecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1/Initializer/zeros*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
в
<decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1/readIdentity7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *
ОБ;*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
_output_shapes
: *
valueB
 *wЙ?*
dtype0
Q
Adam/epsilonConst*
_output_shapes
: *
valueB
 *w╠+2*
dtype0
═
HAdam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/ApplyAdam	ApplyAdam2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
use_nesterov( *
_output_shapes

:@*
use_locking( 
└
FAdam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/ApplyAdam	ApplyAdam0encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias5encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
use_nesterov( *
_output_shapes
:@*
use_locking( 
═
HAdam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/ApplyAdam	ApplyAdam2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonYgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_acc_3*
_output_shapes

:*
use_locking( *
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
use_nesterov( 
└
FAdam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/ApplyAdam	ApplyAdam0encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias5encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonZgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
use_nesterov( *
_output_shapes
:*
use_locking( 
═
HAdam/update_decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/ApplyAdam	ApplyAdam2decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam9decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_acc_3*
_output_shapes

:@*
use_locking( *
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
use_nesterov( 
└
FAdam/update_decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/ApplyAdam	ApplyAdam0decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias5decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
use_locking( *
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
use_nesterov( *
_output_shapes
:@
═
HAdam/update_decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/ApplyAdam	ApplyAdam2decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam9decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonYgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_acc_3*
use_locking( *
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
use_nesterov( *
_output_shapes

:
└
FAdam/update_decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/ApplyAdam	ApplyAdam0decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias5decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonZgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
с
Adam/mulMulbeta1_power/read
Adam/beta1G^Adam/update_decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/ApplyAdamI^Adam/update_decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/ApplyAdamG^Adam/update_decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/ApplyAdamI^Adam/update_decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/ApplyAdamG^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/ApplyAdamG^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/ApplyAdam*
_output_shapes
: *
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias
╗
Adam/AssignAssignbeta1_powerAdam/mul*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
т

Adam/mul_1Mulbeta2_power/read
Adam/beta2G^Adam/update_decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/ApplyAdamI^Adam/update_decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/ApplyAdamG^Adam/update_decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/ApplyAdamI^Adam/update_decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/ApplyAdamG^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/ApplyAdamG^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/ApplyAdam*
_output_shapes
: *
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias
┐
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias
Щ
AdamNoOp^Adam/Assign^Adam/Assign_1G^Adam/update_decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/ApplyAdamI^Adam/update_decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/ApplyAdamG^Adam/update_decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/ApplyAdamI^Adam/update_decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/ApplyAdamG^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/ApplyAdamG^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/ApplyAdamI^Adam/update_encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/ApplyAdam
ъ
initNoOp^beta1_power/Assign^beta2_power/Assign=^decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam/Assign?^decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1/Assign8^decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Assign?^decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/AssignA^decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/Assign:^decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Assign=^decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam/Assign?^decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1/Assign8^decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Assign?^decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam/AssignA^decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1/Assign:^decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Assign=^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam/Assign?^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1/Assign8^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Assign?^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/AssignA^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/Assign:^encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Assign=^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam/Assign?^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1/Assign8^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Assign?^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam/AssignA^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1/Assign:^encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Assign
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
ц
save/SaveV2/tensor_namesConst*О

value═
B╩
Bbeta1_powerBbeta2_powerB0decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasB5decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdamB7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1B2decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdamB9decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1B0decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasB5decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdamB7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1B2decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdamB9decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1B0encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasB5encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdamB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1B2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdamB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1B0encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasB5encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdamB7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1B2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdamB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1*
dtype0*
_output_shapes
:
Ќ
save/SaveV2/shape_and_slicesConst*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
┼
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_power0decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias5decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_12decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam9decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_10decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias5decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_12decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam9decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_10encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias5encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_12encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_10encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias5encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_12encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1*(
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
Х
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*О

value═
B╩
Bbeta1_powerBbeta2_powerB0decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasB5decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdamB7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1B2decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdamB9decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1B0decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasB5decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdamB7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1B2decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdamB9decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1B0encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasB5encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdamB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1B2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdamB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1B0encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasB5encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdamB7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1B2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdamB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1
Е
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ю
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2
┴
save/AssignAssignbeta1_powersave/RestoreV2*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes
: *
use_locking(
┼
save/Assign_1Assignbeta2_powersave/RestoreV2:1*
use_locking(*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes
: 
Ь
save/Assign_2Assign0decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biassave/RestoreV2:2*
use_locking(*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes
:@
з
save/Assign_3Assign5decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adamsave/RestoreV2:3*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
ш
save/Assign_4Assign7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1save/RestoreV2:4*
use_locking(*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes
:@
Ш
save/Assign_5Assign2decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave/RestoreV2:5*
use_locking(*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
validate_shape(*
_output_shapes

:@
ч
save/Assign_6Assign7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adamsave/RestoreV2:6*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
§
save/Assign_7Assign9decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1save/RestoreV2:7*
use_locking(*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
validate_shape(*
_output_shapes

:@
Ь
save/Assign_8Assign0decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biassave/RestoreV2:8*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias
з
save/Assign_9Assign5decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adamsave/RestoreV2:9*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias
э
save/Assign_10Assign7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1save/RestoreV2:10*
use_locking(*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
validate_shape(*
_output_shapes
:
Э
save/Assign_11Assign2decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave/RestoreV2:11*
use_locking(*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
validate_shape(*
_output_shapes

:
§
save/Assign_12Assign7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adamsave/RestoreV2:12*
use_locking(*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
validate_shape(*
_output_shapes

:
 
save/Assign_13Assign9decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1save/RestoreV2:13*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
­
save/Assign_14Assign0encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biassave/RestoreV2:14*
use_locking(*
T0*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes
:@
ш
save/Assign_15Assign5encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adamsave/RestoreV2:15*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
э
save/Assign_16Assign7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1save/RestoreV2:16*
T0*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
Э
save/Assign_17Assign2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave/RestoreV2:17*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
§
save/Assign_18Assign7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adamsave/RestoreV2:18*
_output_shapes

:@*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
validate_shape(
 
save/Assign_19Assign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1save/RestoreV2:19*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
validate_shape(*
_output_shapes

:@
­
save/Assign_20Assign0encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biassave/RestoreV2:20*
use_locking(*
T0*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
validate_shape(*
_output_shapes
:
ш
save/Assign_21Assign5encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adamsave/RestoreV2:21*
use_locking(*
T0*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
validate_shape(*
_output_shapes
:
э
save/Assign_22Assign7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1save/RestoreV2:22*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Э
save/Assign_23Assign2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave/RestoreV2:23*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
§
save/Assign_24Assign7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adamsave/RestoreV2:24*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
validate_shape(*
_output_shapes

:
 
save/Assign_25Assign9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1save/RestoreV2:25*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
validate_shape(*
_output_shapes

:
к
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
R
save_1/ConstConst*
_output_shapes
: *
valueB Bmodel*
dtype0
є
save_1/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_ab47fe1a25fc4f66b3c330356fd7db63/part
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_1/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
ћ
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
х
save_1/SaveV2/tensor_namesConst"/device:CPU:0*О

value═
B╩
Bbeta1_powerBbeta2_powerB0decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasB5decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdamB7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1B2decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdamB9decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1B0decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasB5decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdamB7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1B2decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdamB9decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1B0encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasB5encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdamB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1B2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdamB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1B0encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasB5encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdamB7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1B2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdamB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1*
dtype0*
_output_shapes
:
е
save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Т
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta2_power0decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias5decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_12decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam9decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_10decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias5decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_12decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam9decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_10encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias5encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_12encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_10encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias5encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_12encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1"/device:CPU:0*(
dtypes
2
е
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
▓
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
њ
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*
delete_old_dirs(
Љ
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
И
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*О

value═
B╩
Bbeta1_powerBbeta2_powerB0decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasB5decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdamB7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1B2decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdamB9decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1B0decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasB5decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdamB7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1B2decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdamB9decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1B0encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biasB5encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdamB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1B2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdamB9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1B0encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biasB5encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdamB7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1B2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdamB9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1*
dtype0
Ф
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ц
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2
┼
save_1/AssignAssignbeta1_powersave_1/RestoreV2*
use_locking(*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes
: 
╔
save_1/Assign_1Assignbeta2_powersave_1/RestoreV2:1*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
Ы
save_1/Assign_2Assign0decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biassave_1/RestoreV2:2*
use_locking(*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes
:@
э
save_1/Assign_3Assign5decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adamsave_1/RestoreV2:3*
use_locking(*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes
:@
щ
save_1/Assign_4Assign7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1save_1/RestoreV2:4*
use_locking(*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes
:@
Щ
save_1/Assign_5Assign2decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave_1/RestoreV2:5*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
 
save_1/Assign_6Assign7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adamsave_1/RestoreV2:6*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
Ђ
save_1/Assign_7Assign9decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1save_1/RestoreV2:7*
use_locking(*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
validate_shape(*
_output_shapes

:@
Ы
save_1/Assign_8Assign0decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biassave_1/RestoreV2:8*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
validate_shape(*
_output_shapes
:*
use_locking(
э
save_1/Assign_9Assign5decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adamsave_1/RestoreV2:9*
_output_shapes
:*
use_locking(*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
validate_shape(
ч
save_1/Assign_10Assign7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1save_1/RestoreV2:10*
use_locking(*
T0*C
_class9
75loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
validate_shape(*
_output_shapes
:
Ч
save_1/Assign_11Assign2decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave_1/RestoreV2:11*
_output_shapes

:*
use_locking(*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
validate_shape(
Ђ
save_1/Assign_12Assign7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adamsave_1/RestoreV2:12*
_output_shapes

:*
use_locking(*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
validate_shape(
Ѓ
save_1/Assign_13Assign9decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1save_1/RestoreV2:13*
use_locking(*
T0*E
_class;
97loc:@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
validate_shape(*
_output_shapes

:
З
save_1/Assign_14Assign0encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/biassave_1/RestoreV2:14*
T0*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
щ
save_1/Assign_15Assign5encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adamsave_1/RestoreV2:15*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias
ч
save_1/Assign_16Assign7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1save_1/RestoreV2:16*
_output_shapes
:@*
use_locking(*
T0*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(
Ч
save_1/Assign_17Assign2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave_1/RestoreV2:17*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
validate_shape(*
_output_shapes

:@
Ђ
save_1/Assign_18Assign7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adamsave_1/RestoreV2:18*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
validate_shape(*
_output_shapes

:@
Ѓ
save_1/Assign_19Assign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1save_1/RestoreV2:19*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
validate_shape(*
_output_shapes

:@
З
save_1/Assign_20Assign0encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/biassave_1/RestoreV2:20*
_output_shapes
:*
use_locking(*
T0*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
validate_shape(
щ
save_1/Assign_21Assign5encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adamsave_1/RestoreV2:21*
use_locking(*
T0*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
validate_shape(*
_output_shapes
:
ч
save_1/Assign_22Assign7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1save_1/RestoreV2:22*
use_locking(*
T0*C
_class9
75loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
validate_shape(*
_output_shapes
:
Ч
save_1/Assign_23Assign2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave_1/RestoreV2:23*
_output_shapes

:*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
validate_shape(
Ђ
save_1/Assign_24Assign7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adamsave_1/RestoreV2:24*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
validate_shape(*
_output_shapes

:
Ѓ
save_1/Assign_25Assign9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1save_1/RestoreV2:25*
_output_shapes

:*
use_locking(*
T0*E
_class;
97loc:@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
validate_shape(
■
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"у
trainable_variables¤╠
 
4encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:09encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Assign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read:02Oencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform:08
Ь
2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:07encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Assign7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read:02Dencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/zeros:08
 
4encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:09encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Assign9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read:02Oencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform:08
Ь
2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias:07encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Assign7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read:02Dencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Initializer/zeros:08
 
4decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:09decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Assign9decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read:02Odecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform:08
Ь
2decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:07decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Assign7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read:02Ddecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/zeros:08
 
4decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:09decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Assign9decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read:02Odecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform:08
Ь
2decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias:07decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Assign7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read:02Ddecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Initializer/zeros:08"
train_op

Adam"╣2
	variablesФ2е2
 
4encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:09encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Assign9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read:02Oencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform:08
Ь
2encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:07encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Assign7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read:02Dencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/zeros:08
 
4encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:09encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Assign9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read:02Oencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform:08
Ь
2encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias:07encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Assign7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read:02Dencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Initializer/zeros:08
 
4decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:09decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Assign9decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read:02Odecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform:08
Ь
2decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:07decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Assign7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read:02Ddecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/zeros:08
 
4decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:09decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Assign9decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read:02Odecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform:08
Ь
2decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias:07decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Assign7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read:02Ddecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
ѕ
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam:0>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/Assign>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/read:02Kencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/Initializer/zeros:0
љ
;encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1:0@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/Assign@encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/read:02Mencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/Initializer/zeros:0
ђ
7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam:0<encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam/Assign<encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam/read:02Iencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam/Initializer/zeros:0
ѕ
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1:0>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1/Assign>encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1/read:02Kencoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1/Initializer/zeros:0
ѕ
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam:0>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam/Assign>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam/read:02Kencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam/Initializer/zeros:0
љ
;encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1:0@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1/Assign@encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1/read:02Mencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1/Initializer/zeros:0
ђ
7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam:0<encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam/Assign<encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam/read:02Iencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam/Initializer/zeros:0
ѕ
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1:0>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1/Assign>encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1/read:02Kencoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1/Initializer/zeros:0
ѕ
9decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam:0>decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/Assign>decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/read:02Kdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam/Initializer/zeros:0
љ
;decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1:0@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/Assign@decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/read:02Mdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1/Initializer/zeros:0
ђ
7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam:0<decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam/Assign<decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam/read:02Idecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam/Initializer/zeros:0
ѕ
9decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1:0>decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1/Assign>decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1/read:02Kdecoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1/Initializer/zeros:0
ѕ
9decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam:0>decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam/Assign>decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam/read:02Kdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam/Initializer/zeros:0
љ
;decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1:0@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1/Assign@decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1/read:02Mdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1/Initializer/zeros:0
ђ
7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam:0<decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam/Assign<decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam/read:02Idecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam/Initializer/zeros:0
ѕ
9decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1:0>decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1/Assign>decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1/read:02Kdecoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1/Initializer/zeros:0"░Ж
while_contextЮЖЎЖ
НФ
encoder/rnn/while/while_context *encoder/rnn/while/LoopCond:02encoder/rnn/while/Merge:0:encoder/rnn/while/Identity:0Bencoder/rnn/while/Exit:0Bencoder/rnn/while/Exit_1:0Bencoder/rnn/while/Exit_2:0Bencoder/rnn/while/Exit_3:0Bencoder/rnn/while/Exit_4:0Bencoder/rnn/while/Exit_5:0Bencoder/rnn/while/Exit_6:0Bgradients/f_count_5:0J§д
encoder/random_uniform_1:0
encoder/random_uniform_2:0
encoder/random_uniform_3:0
encoder/random_uniform_5:0
encoder/random_uniform_6:0
encoder/rnn/Minimum:0
encoder/rnn/TensorArray:0
Hencoder/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
encoder/rnn/TensorArray_1:0
7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read:0
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read:0
7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read:0
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read:0
encoder/rnn/strided_slice_1:0
encoder/rnn/while/Enter:0
encoder/rnn/while/Enter_1:0
encoder/rnn/while/Enter_2:0
encoder/rnn/while/Enter_3:0
encoder/rnn/while/Enter_4:0
encoder/rnn/while/Enter_5:0
encoder/rnn/while/Enter_6:0
encoder/rnn/while/Exit:0
encoder/rnn/while/Exit_1:0
encoder/rnn/while/Exit_2:0
encoder/rnn/while/Exit_3:0
encoder/rnn/while/Exit_4:0
encoder/rnn/while/Exit_5:0
encoder/rnn/while/Exit_6:0
encoder/rnn/while/Identity:0
encoder/rnn/while/Identity_1:0
encoder/rnn/while/Identity_2:0
encoder/rnn/while/Identity_3:0
encoder/rnn/while/Identity_4:0
encoder/rnn/while/Identity_5:0
encoder/rnn/while/Identity_6:0
encoder/rnn/while/Less/Enter:0
encoder/rnn/while/Less:0
 encoder/rnn/while/Less_1/Enter:0
encoder/rnn/while/Less_1:0
encoder/rnn/while/LogicalAnd:0
encoder/rnn/while/LoopCond:0
encoder/rnn/while/Merge:0
encoder/rnn/while/Merge:1
encoder/rnn/while/Merge_1:0
encoder/rnn/while/Merge_1:1
encoder/rnn/while/Merge_2:0
encoder/rnn/while/Merge_2:1
encoder/rnn/while/Merge_3:0
encoder/rnn/while/Merge_3:1
encoder/rnn/while/Merge_4:0
encoder/rnn/while/Merge_4:1
encoder/rnn/while/Merge_5:0
encoder/rnn/while/Merge_5:1
encoder/rnn/while/Merge_6:0
encoder/rnn/while/Merge_6:1
!encoder/rnn/while/NextIteration:0
#encoder/rnn/while/NextIteration_1:0
#encoder/rnn/while/NextIteration_2:0
#encoder/rnn/while/NextIteration_3:0
#encoder/rnn/while/NextIteration_4:0
#encoder/rnn/while/NextIteration_5:0
#encoder/rnn/while/NextIteration_6:0
encoder/rnn/while/Switch:0
encoder/rnn/while/Switch:1
encoder/rnn/while/Switch_1:0
encoder/rnn/while/Switch_1:1
encoder/rnn/while/Switch_2:0
encoder/rnn/while/Switch_2:1
encoder/rnn/while/Switch_3:0
encoder/rnn/while/Switch_3:1
encoder/rnn/while/Switch_4:0
encoder/rnn/while/Switch_4:1
encoder/rnn/while/Switch_5:0
encoder/rnn/while/Switch_5:1
encoder/rnn/while/Switch_6:0
encoder/rnn/while/Switch_6:1
+encoder/rnn/while/TensorArrayReadV3/Enter:0
-encoder/rnn/while/TensorArrayReadV3/Enter_1:0
%encoder/rnn/while/TensorArrayReadV3:0
=encoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
7encoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
encoder/rnn/while/add/y:0
encoder/rnn/while/add:0
encoder/rnn/while/add_1/y:0
encoder/rnn/while/add_1:0
3encoder/rnn/while/rnn/multi_rnn_cell/cell_0/Floor:0
5encoder/rnn/while/rnn/multi_rnn_cell/cell_0/Floor_1:0
7encoder/rnn/while/rnn/multi_rnn_cell/cell_0/add/Enter:0
3encoder/rnn/while/rnn/multi_rnn_cell/cell_0/add/x:0
1encoder/rnn/while/rnn/multi_rnn_cell/cell_0/add:0
9encoder/rnn/while/rnn/multi_rnn_cell/cell_0/add_1/Enter:0
5encoder/rnn/while/rnn/multi_rnn_cell/cell_0/add_1/x:0
3encoder/rnn/while/rnn/multi_rnn_cell/cell_0/add_1:0
3encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div/y:0
1encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div:0
5encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1/y:0
3encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1:0
Eencoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter:0
?encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd:0
=encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Const:0
Dencoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter:0
>encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul:0
?encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid:0
Aencoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1:0
Aencoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2:0
<encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh:0
>encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1:0
=encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add/y:0
;encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add:0
=encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1:0
Cencoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat/axis:0
>encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat:0
;encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul:0
=encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1:0
=encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2:0
Gencoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split/split_dim:0
=encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:0
=encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:1
=encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:2
=encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:3
1encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul:0
3encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1:0
3encoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor:0
5encoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor_1:0
5encoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor_2:0
7encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add/Enter:0
3encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add/x:0
1encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add:0
9encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_1/Enter:0
5encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_1/x:0
3encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_1:0
9encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_2/Enter:0
5encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_2/x:0
3encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_2:0
3encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div/y:0
1encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div:0
5encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1/y:0
3encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1:0
5encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2/y:0
3encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2:0
Eencoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter:0
?encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd:0
=encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Const:0
Dencoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter:0
>encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul:0
?encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid:0
Aencoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1:0
Aencoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2:0
<encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh:0
>encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1:0
=encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add/y:0
;encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add:0
=encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1:0
Cencoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat/axis:0
>encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat:0
;encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul:0
=encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1:0
=encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2:0
Gencoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split/split_dim:0
=encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:0
=encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:1
=encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:2
=encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:3
1encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul:0
3encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1:0
3encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2:0
gradients/Add_1/y:0
gradients/Add_1:0
gradients/Merge_2:0
gradients/Merge_2:1
gradients/NextIteration_2:0
gradients/Switch_2:0
gradients/Switch_2:1
^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/Enter:0
dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/StackPushV2:0
^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/f_acc:0
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Shape:0
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/Enter:0
bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/StackPushV2:0
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/f_acc:0
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/Enter:0
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/StackPushV2:0
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/f_acc:0
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Shape:0
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/Enter:0
bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/f_acc:0
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0
jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0
ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2:0
pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1:0
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0
jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape:0
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1:0
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0
lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2:0
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape:0
Sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Shape:0
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/Enter:0
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/Enter_1:0
`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPushV2:0
bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPushV2_1:0
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/f_acc:0
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/f_acc_1:0
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0
jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0
ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2:0
pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1:0
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0
jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/Enter:0
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPushV2:0
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/f_acc:0
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/Enter:0
^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPushV2:0
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/f_acc:0
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape:0
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1:0
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0
jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0
ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2:0
pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1:0
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0
jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/Enter:0
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPushV2:0
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/f_acc:0
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/Enter:0
^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPushV2:0
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/f_acc:0
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape:0
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1:0
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0
lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2:0
ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1:0
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/Enter:0
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPushV2:0
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/f_acc:0
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/Enter:0
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPushV2:0
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/f_acc:0
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape:0
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1:0
^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/Enter:0
dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/StackPushV2:0
^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/f_acc:0
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/Enter:0
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/StackPushV2:0
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/f_acc:0
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/Enter:0
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/StackPushV2:0
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/f_acc:0
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Shape:0
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/Enter:0
bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/StackPushV2:0
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/f_acc:0
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/Enter:0
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/StackPushV2:0
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/f_acc:0
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/Enter:0
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/StackPushV2:0
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/f_acc:0
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Shape:0
^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/Enter:0
dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/StackPushV2:0
^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/f_acc:0
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/Enter:0
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/StackPushV2:0
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/f_acc:0
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Shape:0
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/Enter:0
bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/StackPushV2:0
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/f_acc:0
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/Enter:0
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/StackPushV2:0
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/f_acc:0
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Shape:0
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/Enter:0
bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/f_acc:0
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0
jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0
ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2:0
pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1:0
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0
jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape:0
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1:0
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0
lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2:0
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape:0
Sgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Shape:0
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/Enter:0
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/Enter_1:0
`gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPushV2:0
bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPushV2_1:0
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/f_acc:0
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/f_acc_1:0
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0
jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0
ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2:0
pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1:0
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0
jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/Enter:0
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPushV2:0
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/f_acc:0
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/Enter:0
^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPushV2:0
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/f_acc:0
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape:0
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1:0
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0
jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0
ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2:0
pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1:0
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0
jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/Enter:0
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPushV2:0
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/f_acc:0
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/Enter:0
^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPushV2:0
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/f_acc:0
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape:0
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1:0
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0
lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2:0
ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1:0
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/Enter:0
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPushV2:0
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/f_acc:0
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/Enter:0
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPushV2:0
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/f_acc:0
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape:0
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1:0
^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/Enter:0
dgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/StackPushV2:0
^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/f_acc:0
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/Enter:0
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/StackPushV2:0
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/f_acc:0
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/Enter:0
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/StackPushV2:0
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/f_acc:0
Hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Shape:0
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/Enter:0
bgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/StackPushV2:0
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/f_acc:0
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/Enter:0
Pgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/StackPushV2:0
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/f_acc:0
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/Enter:0
Rgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/StackPushV2:0
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/f_acc:0
Fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Shape:0
gradients/f_count_3:0
gradients/f_count_4:0
gradients/f_count_5:0Z
encoder/rnn/TensorArray:0=encoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0л
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0а
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/f_acc:0Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/Enter:0ю
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/f_acc:0Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/Enter:0ђ
7encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read:0Eencoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter:0W
encoder/random_uniform_2:09encoder/rnn/while/rnn/multi_rnn_cell/cell_0/add_1/Enter:0░
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/f_acc:0Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/Enter:0░
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/f_acc:0Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/Enter:0н
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0░
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/f_acc:0Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/Enter:0░
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/f_acc:0Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/Enter:0╝
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/f_acc:0\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/Enter:0п
jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0И
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/f_acc:0Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/Enter:0л
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0┤
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/f_acc:0Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/Enter:0╝
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/f_acc:0\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/Enter:0U
encoder/random_uniform_3:07encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add/Enter:0┤
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/f_acc:0Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/Enter:0п
jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0г
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/f_acc:0Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/Enter:0ў
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/f_acc:0Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/Enter:0╝
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/f_acc:0\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/Enter:0?
encoder/rnn/strided_slice_1:0encoder/rnn/while/Less/Enter:0╝
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/f_acc_1:0\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/Enter_1:0y
Hencoder/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0-encoder/rnn/while/TensorArrayReadV3/Enter_1:0н
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0н
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0ў
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/f_acc:0Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/Enter:0н
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0л
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/Enter:09
encoder/rnn/Minimum:0 encoder/rnn/while/Less_1/Enter:0ю
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/f_acc:0Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/Enter:0ю
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/f_acc:0Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/Enter:0ђ
7encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read:0Eencoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter:0п
jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0И
Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/f_acc:0Zgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/Enter:0╝
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/f_acc:0\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/Enter:0ў
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/f_acc:0Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/Enter:0└
^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/f_acc:0^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/Enter:0Ђ
9encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read:0Dencoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter:0░
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/f_acc:0Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/Enter:0W
encoder/random_uniform_5:09encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_1/Enter:0н
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0╝
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/f_acc:0\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/Enter:0ю
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/f_acc:0Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/Enter:0ю
Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/f_acc:0Lgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/Enter:0░
Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/f_acc:0Vgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/Enter:0п
jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0ў
Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/f_acc:0Jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/Enter:0г
Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/f_acc:0Tgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/Enter:0J
encoder/rnn/TensorArray_1:0+encoder/rnn/while/TensorArrayReadV3/Enter:0а
Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/f_acc:0Ngradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/Enter:0┤
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/f_acc:0Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/Enter:0┤
Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/f_acc:0Xgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/Enter:0└
^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/f_acc:0^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/Enter:0л
fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0fgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0п
jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0╝
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/f_acc:0\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/Enter:0└
^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/f_acc:0^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/Enter:0W
encoder/random_uniform_6:09encoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_2/Enter:0н
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0п
jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0jgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0U
encoder/random_uniform_1:07encoder/rnn/while/rnn/multi_rnn_cell/cell_0/add/Enter:0н
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0Ђ
9encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read:0Dencoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter:0н
hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0hgradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0╝
\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/f_acc_1:0\gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/Enter_1:0└
^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/f_acc:0^gradients/encoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/Enter:0Rencoder/rnn/while/Enter:0Rencoder/rnn/while/Enter_1:0Rencoder/rnn/while/Enter_2:0Rencoder/rnn/while/Enter_3:0Rencoder/rnn/while/Enter_4:0Rencoder/rnn/while/Enter_5:0Rencoder/rnn/while/Enter_6:0Rgradients/f_count_4:0Zencoder/rnn/strided_slice_1:0
╝Й
decoder/rnn/while/while_context *decoder/rnn/while/LoopCond:02decoder/rnn/while/Merge:0:decoder/rnn/while/Identity:0Bdecoder/rnn/while/Exit:0Bdecoder/rnn/while/Exit_1:0Bdecoder/rnn/while/Exit_2:0Bdecoder/rnn/while/Exit_3:0Bdecoder/rnn/while/Exit_4:0Bdecoder/rnn/while/Exit_5:0Bdecoder/rnn/while/Exit_6:0Bgradients/f_count_2:0JС╣
decoder/random_uniform_1:0
decoder/random_uniform_2:0
decoder/random_uniform_3:0
decoder/random_uniform_5:0
decoder/random_uniform_6:0
decoder/rnn/Minimum:0
decoder/rnn/TensorArray:0
Hdecoder/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
decoder/rnn/TensorArray_1:0
7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read:0
9decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read:0
7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read:0
9decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read:0
decoder/rnn/strided_slice_1:0
decoder/rnn/while/Enter:0
decoder/rnn/while/Enter_1:0
decoder/rnn/while/Enter_2:0
decoder/rnn/while/Enter_3:0
decoder/rnn/while/Enter_4:0
decoder/rnn/while/Enter_5:0
decoder/rnn/while/Enter_6:0
decoder/rnn/while/Exit:0
decoder/rnn/while/Exit_1:0
decoder/rnn/while/Exit_2:0
decoder/rnn/while/Exit_3:0
decoder/rnn/while/Exit_4:0
decoder/rnn/while/Exit_5:0
decoder/rnn/while/Exit_6:0
decoder/rnn/while/Identity:0
decoder/rnn/while/Identity_1:0
decoder/rnn/while/Identity_2:0
decoder/rnn/while/Identity_3:0
decoder/rnn/while/Identity_4:0
decoder/rnn/while/Identity_5:0
decoder/rnn/while/Identity_6:0
decoder/rnn/while/Less/Enter:0
decoder/rnn/while/Less:0
 decoder/rnn/while/Less_1/Enter:0
decoder/rnn/while/Less_1:0
decoder/rnn/while/LogicalAnd:0
decoder/rnn/while/LoopCond:0
decoder/rnn/while/Merge:0
decoder/rnn/while/Merge:1
decoder/rnn/while/Merge_1:0
decoder/rnn/while/Merge_1:1
decoder/rnn/while/Merge_2:0
decoder/rnn/while/Merge_2:1
decoder/rnn/while/Merge_3:0
decoder/rnn/while/Merge_3:1
decoder/rnn/while/Merge_4:0
decoder/rnn/while/Merge_4:1
decoder/rnn/while/Merge_5:0
decoder/rnn/while/Merge_5:1
decoder/rnn/while/Merge_6:0
decoder/rnn/while/Merge_6:1
!decoder/rnn/while/NextIteration:0
#decoder/rnn/while/NextIteration_1:0
#decoder/rnn/while/NextIteration_2:0
#decoder/rnn/while/NextIteration_3:0
#decoder/rnn/while/NextIteration_4:0
#decoder/rnn/while/NextIteration_5:0
#decoder/rnn/while/NextIteration_6:0
decoder/rnn/while/Switch:0
decoder/rnn/while/Switch:1
decoder/rnn/while/Switch_1:0
decoder/rnn/while/Switch_1:1
decoder/rnn/while/Switch_2:0
decoder/rnn/while/Switch_2:1
decoder/rnn/while/Switch_3:0
decoder/rnn/while/Switch_3:1
decoder/rnn/while/Switch_4:0
decoder/rnn/while/Switch_4:1
decoder/rnn/while/Switch_5:0
decoder/rnn/while/Switch_5:1
decoder/rnn/while/Switch_6:0
decoder/rnn/while/Switch_6:1
+decoder/rnn/while/TensorArrayReadV3/Enter:0
-decoder/rnn/while/TensorArrayReadV3/Enter_1:0
%decoder/rnn/while/TensorArrayReadV3:0
=decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
7decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
decoder/rnn/while/add/y:0
decoder/rnn/while/add:0
decoder/rnn/while/add_1/y:0
decoder/rnn/while/add_1:0
3decoder/rnn/while/rnn/multi_rnn_cell/cell_0/Floor:0
5decoder/rnn/while/rnn/multi_rnn_cell/cell_0/Floor_1:0
7decoder/rnn/while/rnn/multi_rnn_cell/cell_0/add/Enter:0
3decoder/rnn/while/rnn/multi_rnn_cell/cell_0/add/x:0
1decoder/rnn/while/rnn/multi_rnn_cell/cell_0/add:0
9decoder/rnn/while/rnn/multi_rnn_cell/cell_0/add_1/Enter:0
5decoder/rnn/while/rnn/multi_rnn_cell/cell_0/add_1/x:0
3decoder/rnn/while/rnn/multi_rnn_cell/cell_0/add_1:0
3decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div/y:0
1decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div:0
5decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1/y:0
3decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1:0
Edecoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter:0
?decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd:0
=decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Const:0
Ddecoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter:0
>decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul:0
?decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid:0
Adecoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1:0
Adecoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2:0
<decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh:0
>decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1:0
=decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add/y:0
;decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add:0
=decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1:0
Cdecoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat/axis:0
>decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat:0
;decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul:0
=decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1:0
=decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2:0
Gdecoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split/split_dim:0
=decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:0
=decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:1
=decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:2
=decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:3
1decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul:0
3decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1:0
3decoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor:0
5decoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor_1:0
5decoder/rnn/while/rnn/multi_rnn_cell/cell_1/Floor_2:0
7decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add/Enter:0
3decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add/x:0
1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add:0
9decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_1/Enter:0
5decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_1/x:0
3decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_1:0
9decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_2/Enter:0
5decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_2/x:0
3decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_2:0
3decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div/y:0
1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div:0
5decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1/y:0
3decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1:0
5decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2/y:0
3decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2:0
Edecoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter:0
?decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd:0
=decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Const:0
Ddecoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter:0
>decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul:0
?decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid:0
Adecoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1:0
Adecoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2:0
<decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh:0
>decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1:0
=decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add/y:0
;decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add:0
=decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1:0
Cdecoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat/axis:0
>decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat:0
;decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul:0
=decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1:0
=decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2:0
Gdecoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split/split_dim:0
=decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:0
=decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:1
=decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:2
=decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:3
1decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul:0
3decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1:0
3decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2:0
gradients/Add/y:0
gradients/Add:0
gradients/Merge:0
gradients/Merge:1
gradients/NextIteration:0
gradients/Switch:0
gradients/Switch:1
^gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
dgradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
^gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/Enter:0
dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/StackPushV2:0
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/f_acc:0
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/Shape:0
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/Enter:0
bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/StackPushV2:0
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/f_acc:0
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/Enter:0
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/StackPushV2:0
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/f_acc:0
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Shape:0
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/Enter:0
bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/f_acc:0
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0
jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0
ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2:0
pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1:0
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0
jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape:0
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/Shape_1:0
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0
lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2:0
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape:0
Sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Shape:0
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/Enter:0
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/Enter_1:0
`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPushV2:0
bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/StackPushV2_1:0
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/f_acc:0
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/f_acc_1:0
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0
jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0
ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2:0
pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1:0
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0
jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/Enter:0
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPushV2:0
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/f_acc:0
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/Enter:0
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPushV2:0
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/f_acc:0
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape:0
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Shape_1:0
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0
jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0
ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2:0
pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1:0
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0
jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/Enter:0
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPushV2:0
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/f_acc:0
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/Enter:0
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPushV2:0
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/f_acc:0
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape:0
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Shape_1:0
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0
lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2:0
ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1:0
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/Enter:0
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPushV2:0
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/f_acc:0
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/Enter:0
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPushV2:0
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/f_acc:0
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape:0
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Shape_1:0
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/Enter:0
dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/StackPushV2:0
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/f_acc:0
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/Enter:0
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/StackPushV2:0
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/f_acc:0
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/Enter:0
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/StackPushV2:0
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/f_acc:0
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Shape:0
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/Enter:0
bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/StackPushV2:0
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/f_acc:0
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/Enter:0
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/StackPushV2:0
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/f_acc:0
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/Enter:0
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/StackPushV2:0
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/f_acc:0
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Shape:0
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/Enter:0
dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/StackPushV2:0
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/f_acc:0
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/Enter:0
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/StackPushV2:0
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/f_acc:0
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Shape:0
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgs/Enter:0
dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgs/StackPushV2:0
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgs/f_acc:0
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/Shape:0
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/Enter:0
bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/StackPushV2:0
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/f_acc:0
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/Enter:0
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/StackPushV2:0
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/f_acc:0
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Shape:0
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/Enter:0
bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/f_acc:0
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0
jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0
ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2:0
pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1:0
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0
jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape:0
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/Shape_1:0
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0
lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2:0
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape:0
Sgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Shape:0
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/Enter:0
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/Enter_1:0
`gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPushV2:0
bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/StackPushV2_1:0
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/f_acc:0
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/f_acc_1:0
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0
jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0
ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2:0
pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1:0
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0
jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/Enter:0
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPushV2:0
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/f_acc:0
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/Enter:0
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPushV2:0
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/f_acc:0
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape:0
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Shape_1:0
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0
jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0
ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2:0
pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1:0
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0
jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/Enter:0
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPushV2:0
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/f_acc:0
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/Enter:0
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPushV2:0
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/f_acc:0
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape:0
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Shape_1:0
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0
lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2:0
ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1:0
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/Enter:0
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPushV2:0
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/f_acc:0
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/Enter:0
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPushV2:0
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/f_acc:0
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape:0
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Shape_1:0
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/Enter:0
dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/StackPushV2:0
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/f_acc:0
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/Enter:0
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/StackPushV2:0
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/f_acc:0
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/Enter:0
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/StackPushV2:0
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/f_acc:0
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Shape:0
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgs/Enter:0
dgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgs/StackPushV2:0
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgs/f_acc:0
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul/Enter:0
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul/StackPushV2:0
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul/f_acc:0
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul_1/Enter:0
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul_1/StackPushV2:0
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul_1/f_acc:0
Hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Shape:0
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/Enter:0
bgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/StackPushV2:0
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/f_acc:0
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/Enter:0
Pgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/StackPushV2:0
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/f_acc:0
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/Enter:0
Rgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/StackPushV2:0
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/f_acc:0
Fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Shape:0
gradients/f_count:0
gradients/f_count_1:0
gradients/f_count_2:0░
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/f_acc:0Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/Enter:0ю
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/f_acc:0Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul_1/Enter:0ю
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/f_acc:0Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul/Enter:0W
decoder/random_uniform_2:09decoder/rnn/while/rnn/multi_rnn_cell/cell_0/add_1/Enter:0░
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/f_acc:0Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/Enter:0╝
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/f_acc:0\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/BroadcastGradientArgs/Enter:0н
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0н
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0л
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0н
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0╝
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/f_acc:0\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/Enter:0ю
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/f_acc:0Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul/Enter:0ю
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/f_acc:0Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul_1/Enter:0U
decoder/random_uniform_3:07decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add/Enter:0J
decoder/rnn/TensorArray_1:0+decoder/rnn/while/TensorArrayReadV3/Enter:0г
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/f_acc:0Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/Enter:0╝
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/f_acc:0\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/BroadcastGradientArgs/Enter:0п
jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0И
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/f_acc:0Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/Enter:0Ђ
9decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read:0Ddecoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter:0а
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul_1/f_acc:0Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul_1/Enter:0л
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0п
jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0н
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0ђ
7decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read:0Edecoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter:09
decoder/rnn/Minimum:0 decoder/rnn/while/Less_1/Enter:0ў
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/f_acc:0Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/Mul/Enter:0┤
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/f_acc:0Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/Enter:0╝
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/f_acc_1:0\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeN/Enter_1:0░
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/f_acc:0Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/Enter:0░
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/f_acc:0Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/Enter:0└
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgs/f_acc:0^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/BroadcastGradientArgs/Enter:0┤
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/f_acc:0Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/Enter:0└
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/f_acc:0^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/BroadcastGradientArgs/Enter:0ю
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/f_acc:0Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/Neg/Enter:0░
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/f_acc:0Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/Enter:0░
Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/f_acc:0Vgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/Enter:0ў
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/f_acc:0Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_grad/Mul/Enter:0ю
Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul/f_acc:0Lgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_2_grad/Mul/Enter:0Ђ
9decoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read:0Ddecoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter:0а
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/f_acc:0Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/Mul_1/Enter:0н
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0л
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0└
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgs/f_acc:0^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_2_grad/BroadcastGradientArgs/Enter:0н
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0└
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/f_acc:0^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_1_grad/BroadcastGradientArgs/Enter:0W
decoder/random_uniform_5:09decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_1/Enter:0╝
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/f_acc:0\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_grad/BroadcastGradientArgs/Enter:0Z
decoder/rnn/TensorArray:0=decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0└
^gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0^gradients/decoder/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0г
Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/f_acc:0Tgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/Enter:0н
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0└
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/f_acc:0^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/mul_1_grad/BroadcastGradientArgs/Enter:0ў
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/f_acc:0Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/div_grad/Neg/Enter:0п
jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0И
Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/f_acc:0Zgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/Enter:0╝
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/f_acc:0\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/BroadcastGradientArgs/Enter:0?
decoder/rnn/strided_slice_1:0decoder/rnn/while/Less/Enter:0л
fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0fgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0п
jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0ў
Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/f_acc:0Jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_grad/Neg/Enter:0╝
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/f_acc:0\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/Enter:0╝
\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/f_acc_1:0\gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeN/Enter_1:0W
decoder/random_uniform_6:09decoder/rnn/while/rnn/multi_rnn_cell/cell_1/add_2/Enter:0ђ
7decoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read:0Edecoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter:0└
^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/f_acc:0^gradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/div_1_grad/BroadcastGradientArgs/Enter:0U
decoder/random_uniform_1:07decoder/rnn/while/rnn/multi_rnn_cell/cell_0/add/Enter:0а
Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/f_acc:0Ngradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/mul_1_grad/Mul_1/Enter:0н
hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0hgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0п
jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0п
jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0jgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0y
Hdecoder/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0-decoder/rnn/while/TensorArrayReadV3/Enter_1:0┤
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/f_acc:0Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/Enter:0┤
Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/f_acc:0Xgradients/decoder/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/Enter:0Rdecoder/rnn/while/Enter:0Rdecoder/rnn/while/Enter_1:0Rdecoder/rnn/while/Enter_2:0Rdecoder/rnn/while/Enter_3:0Rdecoder/rnn/while/Enter_4:0Rdecoder/rnn/while/Enter_5:0Rdecoder/rnn/while/Enter_6:0Rgradients/f_count_1:0Zdecoder/rnn/strided_slice_1:0*Ы
modelУ
.
x1(
Placeholder:0         
0
x2*
Placeholder_1:0         :
state1
encoder/rnn/while/Exit_6:0         ,
y1&
Placeholder_2:0         tensorflow/serving/predict