from monai.utils import first
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from monai.losses import DiceLoss
from tqdm import tqdm


def dice_metric(predicted, target):
    '''
    In this function we take `predicted` and `target` (label) to calculate the dice coeficient then we use it 
    to calculate a metric value for the training and the validation.
    '''
    dice_value = DiceLoss(to_onehot_y = True, sigmoid = True, squared_pred = True)
    value = 1 - dice_value(predicted, target).item() # Loss
    return value

def calculate_weights(val1, val2):
    '''
    In this function we take the number of the background and the forgroud pixels to return the `weights` 
    for the cross entropy loss values.
    '''
    count = np.array([val1, val2])
    summ = count.sum()
    weights = count/summ # Die relative Häufikeit der Klassen 1 und 2
    weights = 1/weights # Die Wahrscheinlichkeit/Schätzung wie genau die Klassen richtig zugeordnet werden 
    summ = weights.sum() # hier werden die Gewichte summiert
    weights = weights/summ # hier werden die Gewichte Normalisiert
    #print(summ, weights)
    return torch.tensor(weights, dtype = torch.float32)

def load_weights(model, weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

def train(model, data_in, loss, optim, max_epochs, model_dir, test_interval = 1, device = torch.device('cpu')):
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_test = []
    save_metric_train = []
    save_metric_test = []
    train_loader, test_loader = data_in
    
    if os.path.exists(os.path.join(model_dir, 'best_metric_model.pth')):
        load_weights(model, os.path.join(model_dir, 'best_metric_model.pth'))
        
    if os.path.exists(os.path.join(model_dir, 'best_metric.txt')):#------------------------------new
        with open(os.path.join(model_dir, 'best_metric.txt'), 'r') as file:
            for metric in file:
                best_metric = metric
                best_metric = float(best_metric)
                print(f'\nBest metric Value loaded from file is: {best_metric}')
    
    for epoch in range(max_epochs):
        print('-' * 10)
        print(f'epoch {epoch + 1}/ {max_epochs}') # aktuelle und maximale epoche
        model.train() # Modell wird in Trainingszustand gesetzt
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train = 0
        for batch_data in train_loader:
            
            train_step += 1
            
            volume = batch_data['vol']
            label = batch_data['seg']
            label = label != 0
            volume, label = (volume.to(device), label.to(device))
            
            optim.zero_grad()
            outputs = model(volume)
            
            train_loss = loss(outputs, label)
            
            train_loss.backward()
            optim.step()
            
            train_epoch_loss += train_loss.item() # gibt den gesamt Verlust der Epoche wieder
            print(
                f'E: {epoch + 1}, Step {train_step} of {len(train_loader)}\n'
                f'Train_loss: {train_loss.item():.4f}' # :.4f gibt die Dezimalstellen nach dem Komma an
            )
            
            train_metric = dice_metric(outputs, label)
            epoch_metric_train += train_metric
            print(f'Train_dice: {train_metric:.4f}')
    
        print('-' * 20)
        
        train_epoch_loss /= train_step # wird der durchschnittlicher Verlust pro Schritt ermittelt
        print(f'Epoch_loss: {train_epoch_loss:.4f}')
        save_loss_train.append(train_epoch_loss)
        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train) # hier werden die Verluste gespeichert
        
        epoch_metric_train /= train_step
        print(f'Epoch_metrics: {epoch_metric_train:.4f}')
        save_metric_train.append(epoch_metric_train)
        np.save(os.path.join(model_dir, 'metric_train.npy'), save_metric_train)
        
        if (epoch + 3) % test_interval == 0:
            
            model.eval()
            with torch.no_grad():
                test_epoch_loss = 0
                test_metric = 0
                epoch_metric_test = 0
                test_step = 0
                
                for test_data in test_loader:
                    
                    test_step += 1
                    
                    test_volume = test_data['vol']
                    test_label = test_data['seg']
                    test_label = test_label != 0
                    test_volume, test_label = (test_volume.to(device), test_label.to(device),)
                    
                    test_outputs = model(test_volume)
                    
                    test_loss = loss(test_outputs, test_label)
                    test_epoch_loss += test_loss.item()
                    test_metric = dice_metric(test_outputs, test_label)
                    epoch_metric_test += test_metric
                
                test_epoch_loss /= test_step
                print(f'test_loss_epoch: {test_epoch_loss:.4f}')
                save_loss_test.append(test_epoch_loss)
                np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_test)
                
                epoch_metric_test /= test_step
                print(f'test_dice_epoch: {epoch_metric_test:.4f}')
                save_metric_test.append(epoch_metric_test)
                np.save(os.path.join(model_dir, 'metric_test.npy'),(save_metric_test))
                
                if epoch_metric_test > best_metric:
                    best_metric = epoch_metric_test
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(model_dir, 'best_metric_model.pth'))
                    with open(os.path.join(model_dir, 'best_metric.txt'), 'w') as file: #-----------------------------------neu
                        file.write(str(best_metric))
                    
                print(
                    f'E: {epoch + 1}, current mean dice: {test_metric:.4f}'
                    f'\nbest mean dice: {best_metric:.4f}'
                    f'at epoch: {best_metric_epoch}'
                )
                
    print(
        f'train completed, best_metric: {best_metric:.4f}'
        f'at epoch: {best_metric_epoch}'
    )
            

def show_patient(data, SLICE_NUMBER = 60, train = True, test = False):
    """
    This function is to show one patient from your datasets, so that you can see if it is okay or you need 
    to change/delete something.

    `data`: this parameter should take the patients from the data loader, which means you need to can the function
    prepare first and apply the transforms that you want after that pass it to this function so that you visualize 
    the patient with the transforms that you want.
    `SLICE_NUMBER`: this parameter will take the slice number that you want to display/show
    `train`: this parameter is to say that you want to display a patient from the training data (by default it is true)
    `test`: this parameter is to say that you want to display a patient from the testing patients.
    """
    check_patient_train, check_patient_test = data
    
    view_train_patient = first(check_patient_train)
    view_test_patient = first(check_patient_test)
    
    if train:
        plt.figure('Visualization Train', (12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f'vol {SLICE_NUMBER}')
        plt.imshow(view_train_patient['vol'][0, 0, :, :, SLICE_NUMBER], cmap = 'gray')
        
        
        plt.subplot(1, 2, 2)
        plt.title(f'seg {SLICE_NUMBER}')
        plt.imshow(view_train_patient['seg'][0, 0, :, :, SLICE_NUMBER])
        plt.show()
    
    if test: 
        plt.figure('Visualization Test', (24, 12))
        plt.subplot(1, 2, 1)
        plt.title(f'vol {SLICE_NUMBER}')
        plt.imshow(view_test_patient['vol'][0, 0, :, :, SLICE_NUMBER])
        
        plt.figure('Visualization Test', (24, 12))
        plt.subplot(1, 2, 2)
        plt.title(f'vol {SLICE_NUMBER}')
        plt.imshow(view_test_patient['seg'][0, 0, :, :, SLICE_NUMBER])
        plt.show()        
                    
                    
def calculate_pixels(data): # Berechnet die Anzahl der Pixel beider Klassen (Hintergrund, Leber) von den Labels
    val = np.zeros((1, 2))

    for batch in tqdm(data):
        batch_label = batch["seg"] != 0
        _, count = np.unique(batch_label, return_counts=True)# gibt eindeutige Elemente als Array und die häufigkeit dieser als zweites Array zurück

        if len(count) == 1:
            count = np.append(count, 0)
        val += count

    #print('The last values:', val)
    return val