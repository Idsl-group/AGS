def attribute(attr_method, x, y, model, **kwargs):
    model.zero_grad()
    kwargs['baselines'] = kwargs['baselines'][:len(x)]
    tensor_attributions = attr_method.attribute(x, target=y, **kwargs)
    return tensor_attributions


