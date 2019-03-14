            # test loss monotonicity
            """
            copy_net = net.Net()
            copy_net.load_state_dict(q_net.state_dict())
            for i in range(0,5):
                new_loss = compute_loss(sample, copy_net, target_net)
                print("Loss after update: {}".format(new_loss.item()))
                copy_net = update_net_parameters(copy_net, gradient)
                if loss < new_loss:
                    print("Warning: gradient descent failed to decrease loss")
                loss = new_loss
            """