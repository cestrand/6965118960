package svm;

import libsvm.svm_node;

public class SVMNode extends svm_node{
	public SVMNode(int index, double value) {
		this.index = index;
		this.value = value;
	}
}
