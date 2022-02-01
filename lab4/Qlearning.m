%% Initialization
%  Initialize the world, Q-table, and hyperparameters
(rng(256));
world = 3;
episode = 1000;

eta = 1;
gamma = 0.9;
epsilon = 0.9;

state = gwinit(world);
actions = [1,2,3,4];

act_prob = ones(size(actions)) / size(actions,2);

Q = zeros(state.ysize, state.xsize, size(actions,2));

%% Training loop
%  Train the agent using the Q-learning algorithm.

for i = 1:episode
    state = gwinit(world);
    i
    while state.isterminal ~= 1
        [action,~] = chooseaction(Q, state.pos(1), state.pos(2), actions, act_prob, epsilon);
        next_state = gwaction(action);
        if next_state.isvalid == 1
            reward = next_state.feedback;
            value = getvalue(Q);
            Q(state.pos(1), state.pos(2), action) = (1-eta)*Q(state.pos(1), state.pos(2), action) + eta*(reward + gamma*value(next_state.pos(1),next_state.pos(2)));
            state = next_state;
        else
            next_pos = next_state.pos + [(action==1) - (action==2),(action==3) - (action==4)]';
            if (next_pos(2) < 1       || ...
                next_pos(2) > next_state.xsize || ...
                next_pos(1) < 1       || ...
                next_pos(1) > next_state.ysize)
              Q(state.pos(1), state.pos(2), action) = -inf;
            else
                reward = -2;
                value = getvalue(Q);
                Q(state.pos(1), state.pos(2), action) = (1-eta)*Q(state.pos(1), state.pos(2), action) + eta*(reward + gamma*value(state.pos(1), state.pos(2)));
            end
        end    
    end
    if i/episode == 0.4
        epsilon = 0.3;
    end
end

figure()
subplot(2,2,1)
imagesc(Q(:,:,1))
subplot(2,2,2)
imagesc(Q(:,:,2))
subplot(2,2,3)
imagesc(Q(:,:,3))
subplot(2,2,4)
imagesc(Q(:,:,4))

gwdraw()

%% Test loop
%  Test the agent (subjectively) by letting it use the optimal policy
%  to traverse the gridworld. Do not update the Q-table when testing.
%  Also, you should not explore when testing, i.e. epsilon=0; always pick
%  the optimal action.
state = gwinit(world);
nbrAction = 0;
policy = getpolicy(Q);
value = getvalue(Q);
figure();
gwdraw();
gwdrawpolicy(policy);

while state.isterminal ~= 1
    action = policy(state.pos(1), state.pos(2));
    next_state = gwaction(action);
    state = next_state;
    nbrAction = nbrAction + 1;
end   

