ŐĘ
Şý
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
ž
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8ť
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
č*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
č*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:č*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:č*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	č
*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	č
*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
č*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
č*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:č*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:č*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	č
*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	č
*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:
*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
č*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
č*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:č*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:č*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	č
*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	č
*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
ö 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ą 
value§ B¤  B 
Ů
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
	variables
regularization_losses
trainable_variables
		keras_api


signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api

iter

beta_1

beta_2
	decay
learning_ratemEmFmGmHvIvJvKvL

0
1
2
3
 

0
1
2
3
­
 layer_regularization_losses
	variables
!layer_metrics
"metrics

#layers
regularization_losses
trainable_variables
$non_trainable_variables
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
%layer_regularization_losses
	variables
&layer_metrics
'metrics

(layers
regularization_losses
trainable_variables
)non_trainable_variables
 
 
 
­
*layer_regularization_losses
	variables
+layer_metrics
,metrics

-layers
regularization_losses
trainable_variables
.non_trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
/layer_regularization_losses
	variables
0layer_metrics
1metrics

2layers
regularization_losses
trainable_variables
3non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 

40
51
62

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	7total
	8count
9	variables
:	keras_api
D
	;total
	<count
=
_fn_kwargs
>	variables
?	keras_api
D
	@total
	Acount
B
_fn_kwargs
C	variables
D	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

70
81

9	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

;0
<1

>	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

@0
A1

C	variables
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
j
serving_default_InputPlaceholder* 
_output_shapes
:
*
dtype0*
shape:

Ĺ
StatefulPartitionedCallStatefulPartitionedCallserving_default_Inputdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin	
2*
Tout
2*
_output_shapes
:	
*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference_signature_wrapper_38948
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
˛
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*$
Tin
2	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*'
f"R 
__inference__traced_save_39363
á
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*#
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__traced_restore_39444Ćŕ
Š
`
B__inference_dropout_layer_call_and_return_conditional_losses_38737

inputs

identity_1S
IdentityIdentityinputs*
T0* 
_output_shapes
:
č2

Identityb

Identity_1IdentityIdentity:output:0*
T0* 
_output_shapes
:
č2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
:
č:H D
 
_output_shapes
:
č
 
_user_specified_nameinputs
¨.

F__inference_MNIST_model_layer_call_and_return_conditional_losses_38989	
input(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityĄ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
č*
dtype02
dense/MatMul/ReadVariableOp}
dense/MatMulMatMulinput#dense/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:č*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2
dense/BiasAddl
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0* 
_output_shapes
:
č2
dense/Sigmoids
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ŹWŻ?2
dropout/dropout/Const
dropout/dropout/MulMuldense/Sigmoid:y:0dropout/dropout/Const:output:0*
T0* 
_output_shapes
:
č2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   č  2
dropout/dropout/ShapeĹ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0* 
_output_shapes
:
č*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *q=>2 
dropout/dropout/GreaterEqual/y×
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0* 
_output_shapes
:
č2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
* 
_output_shapes
:
č2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0* 
_output_shapes
:
č2
dropout/dropout/Mul_1Ś
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	č
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOp
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	
2
dense_1/BiasAddq
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*
_output_shapes
:	
2
dense_1/SoftmaxÁ
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
č*
dtype02-
+dense/kernel/Regularizer/Abs/ReadVariableOpŁ
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2
dense/kernel/Regularizer/Abs
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/ConstŻ
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *˝752 
dense/kernel/Regularizer/mul/x´
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/xą
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/addÇ
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
č*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpŻ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2!
dense/kernel/Regularizer/Square
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 dense/kernel/Regularizer/Const_1¸
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/Sum_1
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *˝752"
 dense/kernel/Regularizer/mul_1/xź
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/mul_1°
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/add_1e
IdentityIdentitydense_1/Softmax:softmax:0*
T0*
_output_shapes
:	
2

Identity"
identityIdentity:output:0*/
_input_shapes
:
:::::O K
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameInput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
é

+__inference_MNIST_model_layer_call_fn_39137

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCallć
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_output_shapes
:	
*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_MNIST_model_layer_call_and_return_conditional_losses_388562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ü
`
'__inference_dropout_layer_call_fn_39222

inputs
identity˘StatefulPartitionedCallŻ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2* 
_output_shapes
:
č* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_387322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0* 
_output_shapes
:
č2

Identity"
identityIdentity:output:0*
_input_shapes
:
č22
StatefulPartitionedCallStatefulPartitionedCall:H D
 
_output_shapes
:
č
 
_user_specified_nameinputs
˝
Ş
B__inference_dense_1_layer_call_and_return_conditional_losses_38761

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	č
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	
2	
BiasAddY
SoftmaxSoftmaxBiasAdd:output:0*
T0*
_output_shapes
:	
2	
Softmax]
IdentityIdentitySoftmax:softmax:0*
T0*
_output_shapes
:	
2

Identity"
identityIdentity:output:0*'
_input_shapes
:
č:::H D
 
_output_shapes
:
č
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ň
z
%__inference_dense_layer_call_fn_39200

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2* 
_output_shapes
:
č*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_387042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0* 
_output_shapes
:
č2

Identity"
identityIdentity:output:0*'
_input_shapes
:
::22
StatefulPartitionedCallStatefulPartitionedCall:H D
 
_output_shapes
:

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
¨

#__inference_signature_wrapper_38948	
input
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCallż
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_output_shapes
:	
*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__wrapped_model_386742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	
2

Identity"
identityIdentity:output:0*/
_input_shapes
:
::::22
StatefulPartitionedCallStatefulPartitionedCall:G C
 
_output_shapes
:


_user_specified_nameInput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
é

+__inference_MNIST_model_layer_call_fn_39150

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCallć
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_output_shapes
:	
*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_MNIST_model_layer_call_and_return_conditional_losses_388992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
óe
Ă
!__inference__traced_restore_39444
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate
assignvariableop_9_total
assignvariableop_10_count
assignvariableop_11_total_1
assignvariableop_12_count_1
assignvariableop_13_total_2
assignvariableop_14_count_2+
'assignvariableop_15_adam_dense_kernel_m)
%assignvariableop_16_adam_dense_bias_m-
)assignvariableop_17_adam_dense_1_kernel_m+
'assignvariableop_18_adam_dense_1_bias_m+
'assignvariableop_19_adam_dense_kernel_v)
%assignvariableop_20_adam_dense_bias_v-
)assignvariableop_21_adam_dense_1_kernel_v+
'assignvariableop_22_adam_dense_1_bias_v
identity_24˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_18˘AssignVariableOp_19˘AssignVariableOp_2˘AssignVariableOp_20˘AssignVariableOp_21˘AssignVariableOp_22˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9˘	RestoreV2˘RestoreV2_1Ś
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*˛
value¨BĽB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesź
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0	*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_2Identity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_2Identity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15 
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_dense_kernel_mIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16
AssignVariableOp_16AssignVariableOp%assignvariableop_16_adam_dense_bias_mIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17˘
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_1_kernel_mIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18 
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_1_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19 
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_dense_kernel_vIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_dense_bias_vIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21˘
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_1_kernel_vIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22 
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_1_bias_vIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22¨
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesÄ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpŘ
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_23ĺ
Identity_24IdentityIdentity_23:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_24"#
identity_24Identity_24:output:0*q
_input_shapes`
^: :::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
˝
Ş
B__inference_dense_1_layer_call_and_return_conditional_losses_39238

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	č
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	
2	
BiasAddY
SoftmaxSoftmaxBiasAdd:output:0*
T0*
_output_shapes
:	
2	
Softmax]
IdentityIdentitySoftmax:softmax:0*
T0*
_output_shapes
:	
2

Identity"
identityIdentity:output:0*'
_input_shapes
:
č:::H D
 
_output_shapes
:
č
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ť.

F__inference_MNIST_model_layer_call_and_return_conditional_losses_39090

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityĄ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
č*
dtype02
dense/MatMul/ReadVariableOp~
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:č*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2
dense/BiasAddl
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0* 
_output_shapes
:
č2
dense/Sigmoids
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ŹWŻ?2
dropout/dropout/Const
dropout/dropout/MulMuldense/Sigmoid:y:0dropout/dropout/Const:output:0*
T0* 
_output_shapes
:
č2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   č  2
dropout/dropout/ShapeĹ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0* 
_output_shapes
:
č*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *q=>2 
dropout/dropout/GreaterEqual/y×
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0* 
_output_shapes
:
č2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
* 
_output_shapes
:
č2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0* 
_output_shapes
:
č2
dropout/dropout/Mul_1Ś
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	č
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOp
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	
2
dense_1/BiasAddq
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*
_output_shapes
:	
2
dense_1/SoftmaxÁ
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
č*
dtype02-
+dense/kernel/Regularizer/Abs/ReadVariableOpŁ
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2
dense/kernel/Regularizer/Abs
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/ConstŻ
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *˝752 
dense/kernel/Regularizer/mul/x´
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/xą
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/addÇ
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
č*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpŻ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2!
dense/kernel/Regularizer/Square
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 dense/kernel/Regularizer/Const_1¸
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/Sum_1
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *˝752"
 dense/kernel/Regularizer/mul_1/xź
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/mul_1°
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/add_1e
IdentityIdentitydense_1/Softmax:softmax:0*
T0*
_output_shapes
:	
2

Identity"
identityIdentity:output:0*/
_input_shapes
:
:::::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
´"
ď
F__inference_MNIST_model_layer_call_and_return_conditional_losses_38899

inputs
dense_38872
dense_38874
dense_1_38878
dense_1_38880
identity˘dense/StatefulPartitionedCall˘dense_1/StatefulPartitionedCallŮ
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_38872dense_38874*
Tin
2*
Tout
2* 
_output_shapes
:
č*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_387042
dense/StatefulPartitionedCallÇ
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2* 
_output_shapes
:
č* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_387372
dropout/PartitionedCallü
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_38878dense_1_38880*
Tin
2*
Tout
2*
_output_shapes
:	
*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_387612!
dense_1/StatefulPartitionedCall¨
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_38872* 
_output_shapes
:
č*
dtype02-
+dense/kernel/Regularizer/Abs/ReadVariableOpŁ
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2
dense/kernel/Regularizer/Abs
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/ConstŻ
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *˝752 
dense/kernel/Regularizer/mul/x´
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/xą
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/addŽ
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_38872* 
_output_shapes
:
č*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpŻ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2!
dense/kernel/Regularizer/Square
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 dense/kernel/Regularizer/Const_1¸
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/Sum_1
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *˝752"
 dense/kernel/Regularizer/mul_1/xź
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/mul_1°
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/add_1ś
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*
_output_shapes
:	
2

Identity"
identityIdentity:output:0*/
_input_shapes
:
::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Š
`
B__inference_dropout_layer_call_and_return_conditional_losses_39217

inputs

identity_1S
IdentityIdentityinputs*
T0* 
_output_shapes
:
č2

Identityb

Identity_1IdentityIdentity:output:0*
T0* 
_output_shapes
:
č2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
:
č:H D
 
_output_shapes
:
č
 
_user_specified_nameinputs
§
¨
@__inference_dense_layer_call_and_return_conditional_losses_39191

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
č*
dtype02
MatMul/ReadVariableOpl
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:č*
dtype02
BiasAdd/ReadVariableOpz
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2	
BiasAddZ
SigmoidSigmoidBiasAdd:output:0*
T0* 
_output_shapes
:
č2	
Sigmoidť
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
č*
dtype02-
+dense/kernel/Regularizer/Abs/ReadVariableOpŁ
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2
dense/kernel/Regularizer/Abs
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/ConstŻ
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *˝752 
dense/kernel/Regularizer/mul/x´
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/xą
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/addÁ
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
č*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpŻ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2!
dense/kernel/Regularizer/Square
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 dense/kernel/Regularizer/Const_1¸
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/Sum_1
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *˝752"
 dense/kernel/Regularizer/mul_1/xź
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/mul_1°
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/add_1X
IdentityIdentitySigmoid:y:0*
T0* 
_output_shapes
:
č2

Identity"
identityIdentity:output:0*'
_input_shapes
:
:::H D
 
_output_shapes
:

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ć

+__inference_MNIST_model_layer_call_fn_39049	
input
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCallĺ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_output_shapes
:	
*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_MNIST_model_layer_call_and_return_conditional_losses_388992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameInput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
á

a
B__inference_dropout_layer_call_and_return_conditional_losses_38732

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ŹWŻ?2
dropout/Constl
dropout/MulMulinputsdropout/Const:output:0*
T0* 
_output_shapes
:
č2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   č  2
dropout/Shape­
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0* 
_output_shapes
:
č*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *q=>2
dropout/GreaterEqual/yˇ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0* 
_output_shapes
:
č2
dropout/GreaterEqualx
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
* 
_output_shapes
:
č2
dropout/Casts
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0* 
_output_shapes
:
č2
dropout/Mul_1^
IdentityIdentitydropout/Mul_1:z:0*
T0* 
_output_shapes
:
č2

Identity"
identityIdentity:output:0*
_input_shapes
:
č:H D
 
_output_shapes
:
č
 
_user_specified_nameinputs
ć

+__inference_MNIST_model_layer_call_fn_39036	
input
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCallĺ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_output_shapes
:	
*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_MNIST_model_layer_call_and_return_conditional_losses_388562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameInput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ü#

F__inference_MNIST_model_layer_call_and_return_conditional_losses_38856

inputs
dense_38829
dense_38831
dense_1_38835
dense_1_38837
identity˘dense/StatefulPartitionedCall˘dense_1/StatefulPartitionedCall˘dropout/StatefulPartitionedCallŮ
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_38829dense_38831*
Tin
2*
Tout
2* 
_output_shapes
:
č*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_387042
dense/StatefulPartitionedCallß
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2* 
_output_shapes
:
č* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_387322!
dropout/StatefulPartitionedCall
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_38835dense_1_38837*
Tin
2*
Tout
2*
_output_shapes
:	
*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_387612!
dense_1/StatefulPartitionedCall¨
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_38829* 
_output_shapes
:
č*
dtype02-
+dense/kernel/Regularizer/Abs/ReadVariableOpŁ
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2
dense/kernel/Regularizer/Abs
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/ConstŻ
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *˝752 
dense/kernel/Regularizer/mul/x´
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/xą
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/addŽ
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_38829* 
_output_shapes
:
č*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpŻ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2!
dense/kernel/Regularizer/Square
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 dense/kernel/Regularizer/Const_1¸
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/Sum_1
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *˝752"
 dense/kernel/Regularizer/mul_1/xź
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/mul_1°
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/add_1Ř
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*
T0*
_output_shapes
:	
2

Identity"
identityIdentity:output:0*/
_input_shapes
:
::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ô
|
'__inference_dense_1_layer_call_fn_39247

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallČ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_output_shapes
:	
*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_387612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	
2

Identity"
identityIdentity:output:0*'
_input_shapes
:
č::22
StatefulPartitionedCallStatefulPartitionedCall:H D
 
_output_shapes
:
č
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ż
h
__inference_loss_fn_0_392678
4dense_kernel_regularizer_abs_readvariableop_resource
identityŃ
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp4dense_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
č*
dtype02-
+dense/kernel/Regularizer/Abs/ReadVariableOpŁ
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2
dense/kernel/Regularizer/Abs
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/ConstŻ
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *˝752 
dense/kernel/Regularizer/mul/x´
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/xą
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add×
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4dense_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
č*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpŻ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2!
dense/kernel/Regularizer/Square
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 dense/kernel/Regularizer/Const_1¸
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/Sum_1
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *˝752"
 dense/kernel/Regularizer/mul_1/xź
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/mul_1°
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/add_1e
IdentityIdentity"dense/kernel/Regularizer/add_1:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
˝%

F__inference_MNIST_model_layer_call_and_return_conditional_losses_39023	
input(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityĄ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
č*
dtype02
dense/MatMul/ReadVariableOp}
dense/MatMulMatMulinput#dense/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:č*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2
dense/BiasAddl
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0* 
_output_shapes
:
č2
dense/Sigmoidn
dropout/IdentityIdentitydense/Sigmoid:y:0*
T0* 
_output_shapes
:
č2
dropout/IdentityŚ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	č
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOp
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	
2
dense_1/BiasAddq
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*
_output_shapes
:	
2
dense_1/SoftmaxÁ
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
č*
dtype02-
+dense/kernel/Regularizer/Abs/ReadVariableOpŁ
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2
dense/kernel/Regularizer/Abs
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/ConstŻ
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *˝752 
dense/kernel/Regularizer/mul/x´
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/xą
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/addÇ
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
č*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpŻ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2!
dense/kernel/Regularizer/Square
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 dense/kernel/Regularizer/Const_1¸
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/Sum_1
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *˝752"
 dense/kernel/Regularizer/mul_1/xź
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/mul_1°
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/add_1e
IdentityIdentitydense_1/Softmax:softmax:0*
T0*
_output_shapes
:	
2

Identity"
identityIdentity:output:0*/
_input_shapes
:
:::::O K
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameInput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


 __inference__wrapped_model_38674	
input4
0mnist_model_dense_matmul_readvariableop_resource5
1mnist_model_dense_biasadd_readvariableop_resource6
2mnist_model_dense_1_matmul_readvariableop_resource7
3mnist_model_dense_1_biasadd_readvariableop_resource
identityĹ
'MNIST_model/dense/MatMul/ReadVariableOpReadVariableOp0mnist_model_dense_matmul_readvariableop_resource* 
_output_shapes
:
č*
dtype02)
'MNIST_model/dense/MatMul/ReadVariableOpĄ
MNIST_model/dense/MatMulMatMulinput/MNIST_model/dense/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2
MNIST_model/dense/MatMulĂ
(MNIST_model/dense/BiasAdd/ReadVariableOpReadVariableOp1mnist_model_dense_biasadd_readvariableop_resource*
_output_shapes	
:č*
dtype02*
(MNIST_model/dense/BiasAdd/ReadVariableOpÂ
MNIST_model/dense/BiasAddBiasAdd"MNIST_model/dense/MatMul:product:00MNIST_model/dense/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2
MNIST_model/dense/BiasAdd
MNIST_model/dense/SigmoidSigmoid"MNIST_model/dense/BiasAdd:output:0*
T0* 
_output_shapes
:
č2
MNIST_model/dense/Sigmoid
MNIST_model/dropout/IdentityIdentityMNIST_model/dense/Sigmoid:y:0*
T0* 
_output_shapes
:
č2
MNIST_model/dropout/IdentityĘ
)MNIST_model/dense_1/MatMul/ReadVariableOpReadVariableOp2mnist_model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	č
*
dtype02+
)MNIST_model/dense_1/MatMul/ReadVariableOpĆ
MNIST_model/dense_1/MatMulMatMul%MNIST_model/dropout/Identity:output:01MNIST_model/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
2
MNIST_model/dense_1/MatMulČ
*MNIST_model/dense_1/BiasAdd/ReadVariableOpReadVariableOp3mnist_model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02,
*MNIST_model/dense_1/BiasAdd/ReadVariableOpÉ
MNIST_model/dense_1/BiasAddBiasAdd$MNIST_model/dense_1/MatMul:product:02MNIST_model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	
2
MNIST_model/dense_1/BiasAdd
MNIST_model/dense_1/SoftmaxSoftmax$MNIST_model/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:	
2
MNIST_model/dense_1/Softmaxq
IdentityIdentity%MNIST_model/dense_1/Softmax:softmax:0*
T0*
_output_shapes
:	
2

Identity"
identityIdentity:output:0*/
_input_shapes
:
:::::O K
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameInput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ŕ%

F__inference_MNIST_model_layer_call_and_return_conditional_losses_39124

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityĄ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
č*
dtype02
dense/MatMul/ReadVariableOp~
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:č*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2
dense/BiasAddl
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0* 
_output_shapes
:
č2
dense/Sigmoidn
dropout/IdentityIdentitydense/Sigmoid:y:0*
T0* 
_output_shapes
:
č2
dropout/IdentityŚ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	č
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOp
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	
2
dense_1/BiasAddq
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*
_output_shapes
:	
2
dense_1/SoftmaxÁ
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
č*
dtype02-
+dense/kernel/Regularizer/Abs/ReadVariableOpŁ
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2
dense/kernel/Regularizer/Abs
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/ConstŻ
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *˝752 
dense/kernel/Regularizer/mul/x´
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/xą
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/addÇ
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
č*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpŻ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2!
dense/kernel/Regularizer/Square
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 dense/kernel/Regularizer/Const_1¸
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/Sum_1
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *˝752"
 dense/kernel/Regularizer/mul_1/xź
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/mul_1°
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/add_1e
IdentityIdentitydense_1/Softmax:softmax:0*
T0*
_output_shapes
:	
2

Identity"
identityIdentity:output:0*/
_input_shapes
:
:::::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ŕ<
Ą	
__inference__traced_save_39363
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_1_const

identity_1˘MergeV2Checkpoints˘SaveV2˘SaveV2_1
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_4c5264d7b19b4072b01289d2dd389f3b/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*˛
value¨BĽB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesś
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *%
dtypes
2	2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardŹ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1˘
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesĎ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1ă
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesŹ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: :
č:č:	č
:
: : : : : : : : : : : :
č:č:	č
:
:
č:č:	č
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
č:!

_output_shapes	
:č:%!

_output_shapes
:	č
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
č:!

_output_shapes	
:č:%!

_output_shapes
:	č
: 

_output_shapes
:
:&"
 
_output_shapes
:
č:!

_output_shapes	
:č:%!

_output_shapes
:	č
: 

_output_shapes
:
:

_output_shapes
: 
Đ
C
'__inference_dropout_layer_call_fn_39227

inputs
identity
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2* 
_output_shapes
:
č* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_387372
PartitionedCalle
IdentityIdentityPartitionedCall:output:0*
T0* 
_output_shapes
:
č2

Identity"
identityIdentity:output:0*
_input_shapes
:
č:H D
 
_output_shapes
:
č
 
_user_specified_nameinputs
á

a
B__inference_dropout_layer_call_and_return_conditional_losses_39212

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ŹWŻ?2
dropout/Constl
dropout/MulMulinputsdropout/Const:output:0*
T0* 
_output_shapes
:
č2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   č  2
dropout/Shape­
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0* 
_output_shapes
:
č*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *q=>2
dropout/GreaterEqual/yˇ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0* 
_output_shapes
:
č2
dropout/GreaterEqualx
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
* 
_output_shapes
:
č2
dropout/Casts
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0* 
_output_shapes
:
č2
dropout/Mul_1^
IdentityIdentitydropout/Mul_1:z:0*
T0* 
_output_shapes
:
č2

Identity"
identityIdentity:output:0*
_input_shapes
:
č:H D
 
_output_shapes
:
č
 
_user_specified_nameinputs
§
¨
@__inference_dense_layer_call_and_return_conditional_losses_38704

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
č*
dtype02
MatMul/ReadVariableOpl
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:č*
dtype02
BiasAdd/ReadVariableOpz
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2	
BiasAddZ
SigmoidSigmoidBiasAdd:output:0*
T0* 
_output_shapes
:
č2	
Sigmoidť
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
č*
dtype02-
+dense/kernel/Regularizer/Abs/ReadVariableOpŁ
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2
dense/kernel/Regularizer/Abs
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/ConstŻ
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *˝752 
dense/kernel/Regularizer/mul/x´
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/xą
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/addÁ
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
č*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpŻ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
č2!
dense/kernel/Regularizer/Square
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 dense/kernel/Regularizer/Const_1¸
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/Sum_1
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *˝752"
 dense/kernel/Regularizer/mul_1/xź
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/mul_1°
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/add_1X
IdentityIdentitySigmoid:y:0*
T0* 
_output_shapes
:
č2

Identity"
identityIdentity:output:0*'
_input_shapes
:
:::H D
 
_output_shapes
:

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: "ŻL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
0
Input'
serving_default_Input:0
3
dense_1(
StatefulPartitionedCall:0	
tensorflow/serving/predict:ź
Á$
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
	variables
regularization_losses
trainable_variables
		keras_api


signatures
M_default_save_signature
N__call__
*O&call_and_return_all_conditional_losses""
_tf_keras_modelô!{"class_name": "Model", "name": "MNIST_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "MNIST_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [150, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}, "name": "Input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1000, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999974752427e-07, "l2": 9.999999974752427e-07}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["Input", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.27, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}], "input_layers": [["Input", 0, 0]], "output_layers": [["dense_1", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [150, 784]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "MNIST_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [150, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}, "name": "Input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1000, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999974752427e-07, "l2": 9.999999974752427e-07}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["Input", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.27, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}], "input_layers": [["Input", 0, 0]], "output_layers": [["dense_1", 0, 0]]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": false}}, "metrics": [{"class_name": "SparseTopKCategoricalAccuracy", "config": {"name": "sparse_top_k_categorical_accuracy", "dtype": "float32", "k": 1}}, {"class_name": "SparseCategoricalCrossentropy", "config": {"name": "sparse_categorical_crossentropy", "dtype": "float32", "from_logits": false, "axis": -1}}], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.019999999552965164, "decay": 0.0, "beta_1": 0.9900000095367432, "beta_2": 0.9900000095367432, "epsilon": 1e-07, "amsgrad": false}}}}
ç"ä
_tf_keras_input_layerÄ{"class_name": "InputLayer", "name": "Input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [150, 784]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [150, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}}
§

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"
_tf_keras_layerč{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1000, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999974752427e-07, "l2": 9.999999974752427e-07}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [150, 784]}}
ż
	variables
regularization_losses
trainable_variables
	keras_api
R__call__
*S&call_and_return_all_conditional_losses"°
_tf_keras_layer{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.27, "noise_shape": null, "seed": null}}
Ó

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
T__call__
*U&call_and_return_all_conditional_losses"Ž
_tf_keras_layer{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1000}}}, "build_input_shape": {"class_name": "TensorShape", "items": [150, 1000]}}

iter

beta_1

beta_2
	decay
learning_ratemEmFmGmHvIvJvKvL"
	optimizer
<
0
1
2
3"
trackable_list_wrapper
'
V0"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
Ę
 layer_regularization_losses
	variables
!layer_metrics
"metrics

#layers
regularization_losses
trainable_variables
$non_trainable_variables
N__call__
M_default_save_signature
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
,
Wserving_default"
signature_map
 :
č2dense/kernel
:č2
dense/bias
.
0
1"
trackable_list_wrapper
'
V0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
%layer_regularization_losses
	variables
&layer_metrics
'metrics

(layers
regularization_losses
trainable_variables
)non_trainable_variables
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
*layer_regularization_losses
	variables
+layer_metrics
,metrics

-layers
regularization_losses
trainable_variables
.non_trainable_variables
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
!:	č
2dense_1/kernel
:
2dense_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
/layer_regularization_losses
	variables
0layer_metrics
1metrics

2layers
regularization_losses
trainable_variables
3non_trainable_variables
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
40
51
62"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
'
V0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ť
	7total
	8count
9	variables
:	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
§
	;total
	<count
=
_fn_kwargs
>	variables
?	keras_api"ŕ
_tf_keras_metricĹ{"class_name": "SparseTopKCategoricalAccuracy", "name": "sparse_top_k_categorical_accuracy", "dtype": "float32", "config": {"name": "sparse_top_k_categorical_accuracy", "dtype": "float32", "k": 1}}
˝
	@total
	Acount
B
_fn_kwargs
C	variables
D	keras_api"ö
_tf_keras_metricŰ{"class_name": "SparseCategoricalCrossentropy", "name": "sparse_categorical_crossentropy", "dtype": "float32", "config": {"name": "sparse_categorical_crossentropy", "dtype": "float32", "from_logits": false, "axis": -1}}
:  (2total
:  (2count
.
70
81"
trackable_list_wrapper
-
9	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
;0
<1"
trackable_list_wrapper
-
>	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
@0
A1"
trackable_list_wrapper
-
C	variables"
_generic_user_object
%:#
č2Adam/dense/kernel/m
:č2Adam/dense/bias/m
&:$	č
2Adam/dense_1/kernel/m
:
2Adam/dense_1/bias/m
%:#
č2Adam/dense/kernel/v
:č2Adam/dense/bias/v
&:$	č
2Adam/dense_1/kernel/v
:
2Adam/dense_1/bias/v
Ý2Ú
 __inference__wrapped_model_38674ľ
˛
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *%˘"
 
Input˙˙˙˙˙˙˙˙˙
ú2÷
+__inference_MNIST_model_layer_call_fn_39036
+__inference_MNIST_model_layer_call_fn_39049
+__inference_MNIST_model_layer_call_fn_39137
+__inference_MNIST_model_layer_call_fn_39150Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ć2ă
F__inference_MNIST_model_layer_call_and_return_conditional_losses_39023
F__inference_MNIST_model_layer_call_and_return_conditional_losses_38989
F__inference_MNIST_model_layer_call_and_return_conditional_losses_39090
F__inference_MNIST_model_layer_call_and_return_conditional_losses_39124Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ď2Ě
%__inference_dense_layer_call_fn_39200˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ę2ç
@__inference_dense_layer_call_and_return_conditional_losses_39191˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
'__inference_dropout_layer_call_fn_39222
'__inference_dropout_layer_call_fn_39227´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Â2ż
B__inference_dropout_layer_call_and_return_conditional_losses_39217
B__inference_dropout_layer_call_and_return_conditional_losses_39212´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ń2Î
'__inference_dense_1_layer_call_fn_39247˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ě2é
B__inference_dense_1_layer_call_and_return_conditional_losses_39238˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
˛2Ż
__inference_loss_fn_0_39267
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
0B.
#__inference_signature_wrapper_38948Input¨
F__inference_MNIST_model_layer_call_and_return_conditional_losses_38989^7˘4
-˘*
 
Input˙˙˙˙˙˙˙˙˙
p

 
Ş "˘

0	

 ¨
F__inference_MNIST_model_layer_call_and_return_conditional_losses_39023^7˘4
-˘*
 
Input˙˙˙˙˙˙˙˙˙
p 

 
Ş "˘

0	

 Š
F__inference_MNIST_model_layer_call_and_return_conditional_losses_39090_8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "˘

0	

 Š
F__inference_MNIST_model_layer_call_and_return_conditional_losses_39124_8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "˘

0	

 
+__inference_MNIST_model_layer_call_fn_39036Q7˘4
-˘*
 
Input˙˙˙˙˙˙˙˙˙
p

 
Ş "	

+__inference_MNIST_model_layer_call_fn_39049Q7˘4
-˘*
 
Input˙˙˙˙˙˙˙˙˙
p 

 
Ş "	

+__inference_MNIST_model_layer_call_fn_39137R8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "	

+__inference_MNIST_model_layer_call_fn_39150R8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "	

 __inference__wrapped_model_38674b/˘,
%˘"
 
Input˙˙˙˙˙˙˙˙˙
Ş ")Ş&
$
dense_1
dense_1	

B__inference_dense_1_layer_call_and_return_conditional_losses_39238M(˘%
˘

inputs
č
Ş "˘

0	

 k
'__inference_dense_1_layer_call_fn_39247@(˘%
˘

inputs
č
Ş "	

@__inference_dense_layer_call_and_return_conditional_losses_39191N(˘%
˘

inputs

Ş "˘

0
č
 j
%__inference_dense_layer_call_fn_39200A(˘%
˘

inputs

Ş "
č
B__inference_dropout_layer_call_and_return_conditional_losses_39212N,˘)
"˘

inputs
č
p
Ş "˘

0
č
 
B__inference_dropout_layer_call_and_return_conditional_losses_39217N,˘)
"˘

inputs
č
p 
Ş "˘

0
č
 l
'__inference_dropout_layer_call_fn_39222A,˘)
"˘

inputs
č
p
Ş "
čl
'__inference_dropout_layer_call_fn_39227A,˘)
"˘

inputs
č
p 
Ş "
č:
__inference_loss_fn_0_39267˘

˘ 
Ş " 
#__inference_signature_wrapper_38948c0˘-
˘ 
&Ş#
!
Input
Input
")Ş&
$
dense_1
dense_1	
