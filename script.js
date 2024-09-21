// Import Matter.js modules
const { Engine, Render, Runner, World, Bodies, Body, Events } = Matter;

// Create engine
const engine = Engine.create();
const world = engine.world;

// Create renderer
const canvas = document.getElementById('world');
const render = Render.create({
    canvas: canvas,
    engine: engine,
    options: {
        width: window.innerWidth * 0.8,
        height: window.innerHeight * 0.6,
        wireframes: false,
        background: '#fafafa'
    }
});

Render.run(render);
const runner = Runner.create();
Runner.run(runner, engine);

// Add ground
const ground = Bodies.rectangle(400, 600, 810, 60, { isStatic: true });
World.add(world, ground);

// Add kart
const kartWidth = 100;
const kartHeight = 20;
const kart = Bodies.rectangle(400, 500, kartWidth, kartHeight, { 
    friction: 0.001, 
    restitution: 0.0 
});
World.add(world, kart);

// Add pole
const poleLength = 150;
const pole = Bodies.rectangle(400, 500 - poleLength / 2, 10, poleLength, {
    friction: 0.001,
    restitution: 0.0
});
World.add(world, pole);

// Constraint between kart and pole
const constraint = Matter.Constraint.create({
    bodyA: kart,
    pointA: { x: 0, y: -kartHeight / 2 },
    bodyB: pole,
    pointB: { x: 0, y: poleLength / 2 },
    length: 0,
    stiffness: 1
});
World.add(world, constraint);

// Keep the kart within the canvas
Events.on(engine, 'beforeUpdate', () => {
    if (kart.position.x < 50 || kart.position.x > render.options.width - 50) {
        // Reset position
        Body.setPosition(kart, { x: 400, y: 500 });
        Body.setVelocity(kart, { x: 0, y: 0 });
        Body.setAngularVelocity(kart, 0);
        
        // Reset pole
        Body.setPosition(pole, { x: 400, y: 500 - poleLength / 2 });
        Body.setVelocity(pole, { x: 0, y: 0 });
        Body.setAngularVelocity(pole, 0);
        
        // Increment episode
        episode++;
        updateInfo();
    }
});

// Handle pole falling
let episode = 0;
let steps = 0;
let cumulativeReward = 0;

// Update Info
function updateInfo() {
    document.getElementById('episode').innerText = episode;
    document.getElementById('steps').innerText = steps;
    document.getElementById('reward').innerText = cumulativeReward.toFixed(2);
}

// Initialize Info
updateInfo();

// Reinforcement Learning Agent using TensorFlow.js
class Agent {
    constructor(stateSize, actionSize) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.learningRate = 0.001;
        this.gamma = 0.95; // Discount rate

        // Define the model
        this.model = this.createModel();
    }

    createModel() {
        const model = tf.sequential();
        model.add(tf.layers.dense({ inputShape: [this.stateSize], units: 24, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 24, activation: 'relu' }));
        model.add(tf.layers.dense({ units: this.actionSize, activation: 'linear' }));
        model.compile({ optimizer: tf.train.adam(this.learningRate), loss: 'meanSquaredError' });
        return model;
    }

    async act(state) {
        const stateTensor = tf.tensor2d([state]);
        const qValues = this.model.predict(stateTensor);
        const actionValues = qValues.dataSync();
        stateTensor.dispose();
        qValues.dispose();
        // Choose the action with the highest Q-value
        const action = actionValues.indexOf(Math.max(...actionValues));
        return action;
    }

    async train(state, action, reward, nextState, done) {
        const target = reward;
        if (!done) {
            const nextQ = this.model.predict(tf.tensor2d([nextState]));
            const nextQVal = nextQ.max(1).dataSync()[0];
            target += this.gamma * nextQVal;
            nextQ.dispose();
        }
        const targetTensor = tf.tensor2d([[target]]);
        const stateTensor = tf.tensor2d([state]);
        const mask = tf.oneHot([action], this.actionSize);
        const withMask = tf.mul(this.model.predict(stateTensor), mask);
        const loss = this.model.trainOnBatch(stateTensor, withMask.add(targetTensor.mul(mask)));
        stateTensor.dispose();
        targetTensor.dispose();
        mask.dispose();
        withMask.dispose();
        return loss;
    }
}

// Initialize Agent
const stateSize = 4; // [kart_velocity, kart_position, pole_angle, pole_angular_velocity]
const actionSize = 2; // [accelerate, steer]
const agent = new Agent(stateSize, actionSize);

// Experience Replay (Optional for better training)
// For simplicity, we'll skip experience replay in this basic implementation

// Hyperparameters
const episodes = 1000;
const maxSteps = 200;
const epsilon = 1.0; // Exploration rate
const epsilonDecay = 0.995;
const epsilonMin = 0.01;

// Main Training Loop
async function train() {
    for (let e = 0; e < episodes; e++) {
        // Reset simulation
        Body.setPosition(kart, { x: 400, y: 500 });
        Body.setVelocity(kart, { x: 0, y: 0 });
        Body.setAngularVelocity(kart, 0);
        
        Body.setPosition(pole, { x: 400, y: 500 - poleLength / 2 });
        Body.setVelocity(pole, { x: 0, y: 0 });
        Body.setAngularVelocity(pole, 0);
        
        steps = 0;
        cumulativeReward = 0;
        updateInfo();
        
        for (let s = 0; s < maxSteps; s++) {
            steps++;
            
            // Get current state
            const state = [
                kart.velocity.x, 
                kart.position.x, 
                pole.angle, 
                pole.angularVelocity
            ];
            
            // Choose action
            let action;
            if (Math.random() < epsilon) {
                // Explore: choose a random action
                action = Math.floor(Math.random() * actionSize);
            } else {
                // Exploit: choose the best action
                action = await agent.act(state);
            }
            
            // Apply action
            // Action 0: Accelerate forward
            // Action 1: Accelerate backward
            // Action 2: Steer left
            // Action 3: Steer right
            // For simplicity, actionSize is 2: [accelerate, steer]
            // We map actions as follows:
            // 0: Accelerate forward
            // 1: Accelerate backward
            // 2: Steer left
            // 3: Steer right
            // However, since actionSize=2, we simplify:
            // 0: Accelerate forward
            // 1: Accelerate backward
            // Steering is kept constant or can be integrated as separate action
            // For this example, we'll focus on acceleration only
            
            // Map action to force
            const forceMagnitude = 0.002;
            if (action === 0) {
                Body.applyForce(kart, kart.position, { x: forceMagnitude, y: 0 });
            } else if (action === 1) {
                Body.applyForce(kart, kart.position, { x: -forceMagnitude, y: 0 });
            }
            
            // Step simulation
            Engine.update(engine, render.options.timeStep);
            
            // Get next state
            const nextState = [
                kart.velocity.x, 
                kart.position.x, 
                pole.angle, 
                pole.angularVelocity
            ];
            
            // Calculate reward
            let reward = 1.0;
            // Penalize large angles
            reward -= Math.abs(pole.angle) * 10;
            
            // Check if pole has fallen
            let done = false;
            if (Math.abs(pole.angle) > Math.PI / 6) { // 30 degrees
                reward = -100;
                done = true;
            }
            
            cumulativeReward += reward;
            updateInfo();
            
            // Train the agent
            await agent.train(state, action, reward, nextState, done);
            
            if (done) {
                console.log(`Episode ${e+1} ended after ${s+1} steps with reward ${cumulativeReward}`);
                break;
            }
        }
        
        // Decay epsilon
        if (epsilon > epsilonMin) {
            epsilon *= epsilonDecay;
        }
    }
    
    console.log('Training completed');
}

// Start training
train();
