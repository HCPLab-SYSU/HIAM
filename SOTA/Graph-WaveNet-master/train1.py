import torch
import numpy as np
import argparse
import time
import util1 as util
import metrics
# import matplotlib.pyplot as plt
from engine1 import trainer
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data',type=str,default='data/hangzhou/OD/OD_26',help='data path')
parser.add_argument('--adjdata',type=str,default='data/hangzhou/graph_hz_conn.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='symnadj',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=4,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=26,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=80,help='number of nodes')
parser.add_argument('--out_dim',type=int,default=26,help='')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.0005,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.1,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=1e-5,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=400,help='')
parser.add_argument('--print_every',type=int,default=1,help='')
parser.add_argument('--save',type=str,default='data/checkpoint',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--seed',type=int,default=777,help='random seed')

parser.add_argument('--exp_base',type=str,default="data",help='log_base_dir')
parser.add_argument('--runs',type=str,default="debug",help='log runs name')

parser.add_argument('--train_type',type=str,default="od",help='od training or do training') 

args = parser.parse_args()

def evaluate(scaler, dataloader, device, engine, logger,type, cat_list):
    results = {}
    y_preds = []
    gt = []
    for iter, (x, y) in enumerate(dataloader[type + '_loader'].get_iterator()):
        test_x = torch.tensor(x, dtype=torch.float, device=device)
        # testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = engine.model(test_x)  # .transpose(1,3)
            # preds = preds.transpose(1,3)
        test_y = torch.tensor(y, dtype=torch.float, device=device)
        y_preds.append(preds.detach().cpu().numpy())
        gt.append(test_y.detach().cpu().numpy())

    res = [] 
    for category in cat_list:
        y_preds = np.concatenate(y_preds, axis=0)  # concat in batch_size dim.
        gt = np.concatenate(gt, axis=0)
        mae_list = []
        mape_net_list = []
        rmse_list = []
        mae_sum = 0

        mape_net_sum = 0
        rmse_sum = 0
        horizon = 4
        for horizon_i in range(horizon):
            y_pred = scaler.inverse_transform(
                y_preds[:, horizon_i, :, :])
            y_pred[y_pred < 0] = 0
            y_truth = gt[:, horizon_i, :, :]
            mae = metrics.masked_mae_np(y_pred, y_truth)
            mape_net = metrics.masked_mape_np(y_pred, y_truth)
            rmse = metrics.masked_rmse_np(y_pred, y_truth)
            mae_sum += mae
            mape_net_sum += mape_net
            rmse_sum += rmse
            mae_list.append(mae)

            mape_net_list.append(mape_net)
            rmse_list.append(rmse)

            msg = "Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE_net: {:.4f}"
            if type=='test':
                logger.info(msg.format(horizon_i + 1, mae, rmse, mape_net))
                res.append(
                    {
                        "MAE": mae,
                        "RMSE": rmse,
                        "MAPE_net": mape_net,
                    }
                )
        results['MAE_' + category] = mae_sum / horizon
        results['RMSE_' + category] = rmse_sum / horizon
        results['MAPE_net_' + category] = mape_net_sum / horizon
    
    if type=='test': 
        logger.info('Evaluation_{}_End:'.format(type))
        return results, res
    else:
        return results

def main():
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    tb_writer, log_dir = util.setup_tensorboard_writer(os.path.join(args.exp_base, args.runs), comment="",
            epochs=args.epochs, lr = args.learning_rate, bn=args.batch_size, nhid=args.nhid, gcn="t" if args.gcn_bool else 'f')

    # load data
    logger = util.get_logger(log_dir, __name__, 'info.log', level='INFO')
    args.log_dir = log_dir
    # device = torch.device(args.device)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    # sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    adj_mx = util.load_adj(args.adjdata, args.adjtype)
    if args.train_type == "od":
        dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    else:
        dataloader = util.load_dataset_do(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.out_dim, args.nhid, 
                    args.dropout, args.learning_rate, args.weight_decay, device, 
                    supports, args.gcn_bool, args.addaptadj,adjinit)


    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []


    best_epoch = 0
    best_val_net_mape = 1e6
    update = {}
    for category in ['od', 'do']:
        update['val_steady_count_'+category] = 0
        update['last_val_mae_'+category] = 1e6
        update['last_val_mape_net_'+category] = 1e6

    global_step = 0
    for i in range(1,args.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.tensor(x, dtype=torch.float, device=device)
            trainy = torch.tensor(y, dtype=torch.float, device=device)

            loss, predicts, gt = engine.train(trainx, trainy)
            train_loss.append(loss)
            tb_writer.add_scalar('train/loss', loss, global_step)
            
            global_step += 1

        if i % args.print_every == 0:
            logger.info(('Epoch:{}').format(i))
            val_result = evaluate(scaler, dataloader, device, engine, logger, 'val', [args.train_type])

            test_res, res = evaluate(scaler, dataloader, device, engine, logger, 'test', [args.train_type])
            for k, v in test_res.items():
                tb_writer.add_scalar(f'metric_test/{k}', v, i) 

            val_category = [args.train_type]
            for k, v in val_result.items():
                tb_writer.add_scalar(f'metric_val/{k}', v, i)
            for category in val_category:
                logger.info('{}:'.format(category))
                logger.info(('val_mae:{}, val_mape_net:{}').format(
                    val_result['MAE_' + category],
                    val_result['MAPE_net_' + category]))
                if val_result['MAPE_net_' + category] < best_val_net_mape:
                    best_val_net_mape = val_result['MAPE_net_' + category]
                    metrics_strs = ['MAE', 'RMSE', 'MAPE_net']
                    best_epoch = i
                    with open(os.path.join(args.log_dir, 'a_res.csv'), 'w') as f:
                        f.write(f"{args.log_dir}_{best_epoch}\n")
                        for met in metrics_strs:
                            f.write(',\n'.join([str(e[met]) for e in res] + [',\n']))
                    torch.save(engine.model.state_dict(), os.path.join(args.log_dir, "best_model.pth"))
                if update['last_val_mae_' + category] > val_result['MAE_' + category]:
                    logger.info('val_mae decreased from {} to {}'.format(
                        update['last_val_mae_' + category],
                        val_result['MAE_' + category]))
                    update['last_val_mae_' + category] = val_result['MAE_' + category]

                if update['last_val_mape_net_' + category] > val_result['MAPE_net_' + category]:
                    logger.info('val_mape_net decreased from {} to {}'.format(
                        update['last_val_mape_net_' + category],
                        val_result['MAPE_net_' + category]))
                    update['last_val_mape_net_' + category] = val_result['MAPE_net_' + category]

        torch.save(engine.model.state_dict(), os.path.join(args.log_dir, "epoch_"+str(i)+".pth"))
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    print("Training finished")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
