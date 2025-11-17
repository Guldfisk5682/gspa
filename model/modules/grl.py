from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor, lambda_):
        ctx.lambda_ = lambda_
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_ = ctx.lambda_
        reversed_grad = grad_output.neg() * lambda_
        return reversed_grad, None


def grad_reverse(x, lambda_):
    return GradientReversalFunction.apply(x, lambda_)
