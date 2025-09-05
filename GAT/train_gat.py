from layers import *
from models import *
from torch.optim import Adam
import sys
from models import GAT
import os
import time
# Add the directory containing utils.py to sys.path
sys.path.append(r"C:\Users\Noman\Desktop\Github\Tools\W_M_I_GNN")
from utils import *

class LoopPhase:
    TRAIN = 0
    VAL = 1
    TEST = 2


def get_main_loop(gat, loss_fn, optimizer, node_features, node_labels, edge_index,
                  train_indices, val_indices, test_indices, patience_period, time_start):

    node_dim = 0  # dimension for nodes

    train_labels = node_labels.index_select(node_dim, train_indices)
    val_labels = node_labels.index_select(node_dim, val_indices)
    test_labels = node_labels.index_select(node_dim, test_indices)

    graph_data = (node_features, edge_index)

    def get_node_indices(phase):
        if phase == LoopPhase.TRAIN:
            return train_indices
        elif phase == LoopPhase.VAL:
            return val_indices
        else:
            return test_indices

    def get_node_labels(phase):
        if phase == LoopPhase.TRAIN:
            return train_labels
        elif phase == LoopPhase.VAL:
            return val_labels
        else:
            return test_labels

    def main_loop(phase, epoch=0):
        global BEST_VAL_ACC, BEST_VAL_LOSS, PATIENCE_CNT

        if phase == LoopPhase.TRAIN:
            gat.train()
        else:
            gat.eval()

        node_indices = get_node_indices(phase)
        gt_node_labels = get_node_labels(phase)

        nodes_unnormalized_scores = gat(graph_data)[0].index_select(node_dim, node_indices)
        loss = loss_fn(nodes_unnormalized_scores, gt_node_labels)

        if phase == LoopPhase.TRAIN:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        class_predictions = torch.argmax(nodes_unnormalized_scores, dim=-1)
        accuracy = torch.sum(torch.eq(class_predictions, gt_node_labels).long()).item() / len(gt_node_labels)

        # Validation logic for early stopping
        if phase == LoopPhase.VAL:
            if accuracy > BEST_VAL_ACC or loss.item() < BEST_VAL_LOSS:
                BEST_VAL_ACC = max(accuracy, BEST_VAL_ACC)
                BEST_VAL_LOSS = min(loss.item(), BEST_VAL_LOSS)
                PATIENCE_CNT = 0
            else:
                PATIENCE_CNT += 1

            if PATIENCE_CNT >= patience_period:
                raise Exception("Stopping training: patience exhausted.")

        elif phase == LoopPhase.TRAIN:
            pass  # you can add optional logging here if needed
        else:  # TEST phase
            return accuracy

    return main_loop




# Make sure these constants are defined
CORA_NUM_INPUT_FEATURES = 1433
CORA_NUM_CLASSES = 7

BINARIES_PATH = os.path.join(os.getcwd(), 'models', 'binaries')
os.makedirs(BINARIES_PATH, exist_ok=True)

BEST_VAL_ACC = 0
BEST_VAL_LOSS = 0
PATIENCE_CNT = 0

NUM_EPOCHS = 1000
LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-4
PATIENCE_PERIOD = 100
SHOULD_TEST = True


def train_gat():
    global BEST_VAL_ACC, BEST_VAL_LOSS, PATIENCE_CNT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = r"C:\Users\Noman\Desktop\Github\Tools\W_M_I_GNN\data_gat\cora"
    edge_index, features, labels, idx_train, idx_val, idx_test = load_data_gat(data_path=data_path, device=device)

    # edge_index, features, labels, idx_train, idx_val, idx_test = load_data_gat("data_gat/cora", device)

    # ----------------------------
    # Hardcoded GAT architecture
    # ----------------------------
    gat = GAT(
        num_of_layers=2,
        num_heads_per_layer=[8, 1],
        num_features_per_layer=[CORA_NUM_INPUT_FEATURES, 8, CORA_NUM_CLASSES],
        add_skip_connection=False,
        bias=True,
        dropout=0.6
    ).to(device)

    # Loss & optimizer
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = Adam(gat.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Main training loop
    main_loop = get_main_loop(
        gat,
        loss_fn,
        optimizer,
        features,
        labels,
        edge_index,
        idx_train,
        idx_val,
        idx_test,
        PATIENCE_PERIOD,
        time.time()
    )

    BEST_VAL_ACC, BEST_VAL_LOSS, PATIENCE_CNT = [0, 0, 0]

    for epoch in range(NUM_EPOCHS):
        # Training phase
        main_loop(phase=LoopPhase.TRAIN, epoch=epoch)

        # Validation phase
        with torch.no_grad():
            try:
                main_loop(phase=LoopPhase.VAL, epoch=epoch)
            except Exception as e:
                print(str(e))
                break

    # Test phase
    if SHOULD_TEST:
        test_acc = main_loop(phase=LoopPhase.TEST)
        print(f'Test accuracy = {test_acc}')

    # Save the trained model
    torch.save(gat.state_dict(), os.path.join(BINARIES_PATH, 'gat_trained.pth'))


train_gat()
