import torch
import torch.utils.data as data


def valid_class_acc(classify_model, valid_data_loader):
    classify_model = classify_model.eval()
    
    indices = valid_data_loader.dataset.get_train_indices()
    new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
    valid_data_loader.batch_sampler.sampler = new_sampler
    
    embeds, targets = next(iter(valid_data_loader))
    
    embeds = embeds.squeeze(1)
    targets = targets.type(torch.LongTensor).to(device)
        
    outputs = classify_model(embeds)
    
    predict_result = outputs.argmax(1)
    size = len(predict_result)
    accuracy = torch.sum(predict_result == targets).item() / size * 100
    
    return accuracy