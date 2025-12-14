using CSV, DataFrames, Random, Printf

# Q-learning model
mutable struct QLearning
    S       # state space (assumed to be 1:nstates)
    A       # action space (assumed to be 1:nactions)
    gamma   # discount factor
    Q       # Q(s,a) - action-value function
    alpha   # learning rate
end

# Q-learning update
function update!(model::QLearning, s, a, r, s′, alpha)
    gamma, Q = model.gamma, model.Q
    # Q[s,a] ← Q[s,a] + alpha * (r + gamma * max_a' Q[s′,a′] − Q[s,a])
    Q[s, a] += alpha * (r + gamma * maximum(Q[s′, :]) - Q[s, a])
    return model
end

# Train Q-learning model to find optimal Q(s,a) function
function train_qlearning(csvfile::String)
    # Read input file and initialize hyperparameters
    df = CSV.read(csvfile, DataFrame)
    alpha = 0.1
    nepochs = 10

    if csvfile == "small.csv"
        nstates = 100
        nactions = 4
        gamma = 0.95
    elseif csvfile == "medium.csv"
        nstates = 50000
        nactions = 7
        gamma = 1.0
    elseif csvfile == "large.csv"
        nstates = 302020
        nactions = 9
        gamma = 0.95
    end

    # Initialize Q(s,a) = 5 for all (s,a) to encourage exploration
    model = QLearning(1:nstates, 1:nactions, gamma, fill(5.0, nstates, nactions), alpha)

    # Find states that appear in the s column (used to detect states that don't)
    nonterminal_states = Set(df.s)

    for epoch in 1:nepochs
        println("Epoch $epoch")

        # Shuffle input data order to avoid sequential bias
        shuffled = df[shuffle(axes(df, 1)), :]

        for row in eachrow(shuffled)
            s = Int(row.s)
            a = Int(row.a)
            r = Float64(row.r)
            sp = Int(row.sp)

            # Modified rewards for medium.csv
            if csvfile == "medium.csv"
                pos = (s - 1) % 500
                r += 0.1 * (pos / 500.0)
            end

            # Non-modified update for non-terminal s'
            if sp in nonterminal_states
                update!(model, s, a, r, sp, alpha)
            else
                # Modified update for terminal s': no next Q
                model.Q[s, a] += alpha * (r - model.Q[s, a])
            end
        end
    end

    return model
end


# Save policy as text in output file
function save_policy(model::QLearning, file_name::String)
    open(file_name, "w") do f
        for s in model.S
            best_action = argmax(model.Q[s, :]) 
            write(f, @sprintf("%d\n", best_action))
        end
    end
end


if length(ARGS) != 2
    error("usage: julia project2.jl <infile>.csv <outfile>.policy")
end

inputfilename = ARGS[1]
outputfilename = ARGS[2]
model = train_qlearning(inputfilename)
save_policy(model, outputfilename)