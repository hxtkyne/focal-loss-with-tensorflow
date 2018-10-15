# coding: utf-8
# author: hxtkyne/dianmao
# description: focal loss for deep learning
# Reference Paper : Focal Loss for Dense Object Detection
# Reference code: https://github.com/ailias/Focal-Loss-implement-on-Tensorflow
import numpy as np
import tensorflow as tf

def focal_loss_sigmoid_on_2_classification(labels, logtis, alpha=0.5, gamma=2):
	"""
	description: 
		基于logtis输出的2分类focal loss计算

	计算公式：
		pt = p if label=1, else pt = 1-p； p表示判定为类别1（正样本）的概率
		focal loss = - alpha * (1-pt) ** (gamma) * log(pt)
	
	Args:
		labels: [batch_size], dtype=int32，值为0或者1
		logits: [batch_size], dtype=float32，输入为logits值
		alpha: 控制样本数目的权重，当正样本数目大于负样本时，alpha<0.5，反之，alpha>0.5。
		gamma：focal loss的超参数
	Returns:
		tensor: [batch_size]
	"""
	y_pred = tf.nn.sigmoid(logits) # 转换成概率值
	labels = tf.to_float(labels) # int -> float

	"""
	if label=1, loss = -alpha * ((1 - y_pred) ** gamma) * tf.log(y_pred)
	if label=0, loss = - (1 - alpha) * (y_pred ** gamma) * tf.log(1 - y_pred)
	alpha=0.5，表示赋予不考虑数目差异，此时权重是一致的
	将上面两个标签整合起来，得到下面统一的公式：
		focal loss = -alpha * (1-p)^gamma * log(p) - (1-apha) * p^gamma * log(1-p)
	"""
	loss = -labels * alpha * ((1 - y_pred) ** gamma) * tf.log(y_pred) \
		-(1 - labels) * (1 - alpha) * (y_pred ** gamma) * tf.log(1 - y_pred)
	return loss


def focal_loss_sigmoid_on_multi_classification(labels, logits, gamma=2):
	"""
	description:
		基于多类别的focal loss计算（其实就是转换成one-hot处理）

	Args:
		labels: [batch_size], dtype=int32，值为0或者1
		logits: [batch_size, num_classes], dtype=float32
		gamma：focal loss的超参数

	Returns:
		tensor: [batch_size]
	"""
	y_pred = tf.nn.softmax(logits, dim=-1) # [batch_size, num_classes]
	labels = tf.to_float(labels) # label example: [0,1,2,3]
	labels = tf.one_hot(labels,depth=y_pred.shape[1]) # [0,1,2,3] -> [[0.,0.,0.,0.], [0.,1.,0.,.0], xxx], dtype=float32

	loss = -labels * ((1 - y_pred) ** gamma) * tf.log(y_pred)
	loss = tf.reduce_sum(L, axis=1)
	return loss


from tensorflow.python.ops import array_ops

def focal_loss_on_object_detection(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
	"""
	focal loss = -alpha * (z-p)^gamma * log(p) - (1-alpha) * p^gamma * log(1-p)
	注（z-p)那一项，因为z是one-hot编码，公式其他部分都正常对应
	
	Args:
		prediction_tensor: [batch_size, num_anchors, num_classes]，one-hot表示
	 	target_tensor: [batch_size, num_anchors, num_classes] one-hot表示
		weights: [batch_size, num_anchors]
		alpha: focal loss超参数
		gamma: focal loss超参数
	Returns:
		loss: 返回loss的tensor常量
	"""
	sigmoid_p = tf.nn.sigmoid(prediction_tensor)
	zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

	# 对于positive prediction，只考虑前景部分的loss，背景loss为0
	# target_tensor > zeros <==> z=1, 所以positive系数 = z - p
	pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

	# 对于negative prediction，只考虑背景部分的loss，前景的为0
	# target_tensor > zeros <==> z=1, 所以negative系数 = 0
	neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)

	per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
		 - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
	
	return tf.reduce_sum(per_entry_cross_ent)


if __name__ == '__main__':
	logits = tf.Variable([-1., 1., -0.5, 0.5])
	labels = tf.Variable([0, 1, 0, 1])
	loss = focal_loss_sigmoid_on_2_classification(labels, logits)

	logits2 = tf.Variable([[-1,1], [1,-1], [-0.5,0.5], [0.5,-0.5]])
	labels2 = tf.Variable([0, 1, 0, 1])
	loss2 = focal_loss_sigmoid_on_2_classification(labels2, logits2)

	prediction_tensor = tf.Variable([[[1.,-1.],[-1.,1.]]])
	target_tensor = tf.Variable([[[1.,0.],[0.,1.]]])
	loss3 = focal_loss_on_object_detection(prediction_tensor, target_tensor)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		loss = sess.run(loss)
		loss2 = sess.run(loss2)
		loss3 = sess.run(loss3)
		# print(sess.run(tf.one_hot(labels, depth=2)).dtype)
		print(loss, loss2, loss3)


