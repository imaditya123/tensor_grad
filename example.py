from UniTensor.tensor_autograd import UniTensor

# Example for using UniTensor
a=UniTensor(10.0)
b=UniTensor(5.0)
c=a/b
p=c.draw_dot()
p.render('gout')

