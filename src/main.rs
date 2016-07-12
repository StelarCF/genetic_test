extern crate rand;
use std::cmp::Ordering;
use rand::distributions::{Exp, Range, IndependentSample};

// Going to assume totally inelastic ground
// As such, a Node's friction is applied when y < 0.01

macro_rules! minmaxrange {
    ($min:expr, $max:expr, $middle:expr, $variance:expr) =>
        (Range::new(if $middle - $variance < $min { $min } else { $middle - $variance },
                    if $middle + $variance > $max { $max } else { $middle + $variance } + 0.01))
}

// Units are in meters, seconds, m/s, N, kg
// (Corresp. to length, time, speed, force, mass)

static mut next_id : i64 = 0;

// Gravity
static GRAVITY_ACCEL : f64 = -9.832;
// Average mutation
static MUTATE_LAMBDA : f64 = 0.1;
// Steps per fitness test, and time (in seconds) per step (commented due to warn dead code)
static FITNESS_STEPS : i64 = 750;
static STEP_SECONDS : f64 = 0.02;
// Statics that determine the minimum and maximum characteristics of
// Nodes:
static MIN_FRICTION : f64 = 0.0;
static MAX_FRICTION : f64 = 1.0;
static MUT_FRICTION : f64 = 0.5; // Mutation factor of friction, multiply the mutation factor by this; these values is generally MAX - MIN

static MIN_RADIUS : f64 = 1.0;
static MAX_RADIUS : f64 = 1.5;
static MUT_RADIUS : f64 = 0.5;

static MIN_MASS : f64 = 1.0;
static MAX_MASS : f64 = 2.0;
static MUT_MASS : f64 = 0.05;

static MIN_X : f64 = -2.0;
static MAX_X : f64 = 2.0;
static MIN_Y : f64 = 0.1;
static MAX_Y : f64 = 3.0;
static MUT_XY : f64 = 0.05;
// Muscles:
static MIN_FORCE : f64 = 0.0;
static MAX_FORCE : f64 = 20.0;
static MUT_FORCE : f64 = 0.5;

static MIN_CYCLE_TIME : f64 = 0.02; // in seconds
static MAX_CYCLE_TIME : f64 = 3.00;
static MUT_CYCLE_TIME : f64 = 0.25;

static MIN_MUSCLE_LENGTH : f64 = 0.1; // minimum both for "contracted" and "extended" (I don't differentiate, since they might swap between themselves)
static MAX_MUSCLE_LENGTH : f64 = 3.0;
static MUT_MUSCLE_LENGTH : f64 = 0.25;

static MUSCLE_DELTA_ACCEPT : f64 = 0.2;
// Creatures:
static MIN_NODES : f64 = 3.0; // We round it up, but it's easier this way for the range thing
static MAX_NODES : f64 = 10.9;
static MUT_NODES : f64 = 1.0;

trait Mutate : Clone {
    fn mutate(&self) -> Self;
}

impl<T> Mutate for Vec<T> where T : Mutate {
    fn mutate(&self) -> Vec<T> {
        self.iter().map(|e : &T| e.mutate()).collect()
    }
}

#[derive(Clone)]
struct Node {
    friction: f64,
    radius: f64,
    mass: f64,
    x_gen: f64,
    y_gen: f64,
    x: f64,
    y: f64,
    vx: f64,
    vy: f64
}

impl Node {
    fn new() -> Node {
        let mut rng = rand::thread_rng();
        let friction_range = Range::new(MIN_FRICTION, MAX_FRICTION);
        let radius_range = Range::new(MIN_RADIUS, MAX_RADIUS);
        let mass_range = Range::new(MIN_MASS, MAX_MASS);
        let x_range = Range::new(MIN_X, MAX_X);
        let y_range = Range::new(MIN_Y, MAX_Y);
        Node {
            friction: friction_range.ind_sample(&mut rng),
            radius: radius_range.ind_sample(&mut rng),
            mass: mass_range.ind_sample(&mut rng),
            x_gen: x_range.ind_sample(&mut rng),
            y_gen: y_range.ind_sample(&mut rng),
            x: 0.0,
            y: 0.0,
            vx: 0.0,
            vy: 0.0
        }
    }
    fn ground_distance(&self) -> f64 {
        self.y - self.radius
    }
    fn put_on_ground(&mut self) {
        let ground_dist = self.ground_distance();
        if ground_dist < 0.0 {
            self.y -= ground_dist;
        }
    }
    fn reset(&mut self) {
        self.x = self.x_gen;
        self.y = self.y_gen;
        self.vx = 0.0;
        self.vy = 0.0;
    }
}

impl Mutate for Node {
    fn mutate(&self) -> Node {
        let mut rng = rand::thread_rng();
        let distribution = Exp::new(1.0 / MUTATE_LAMBDA);
        let mutation_factor = distribution.ind_sample(&mut rng);
        let friction_range = minmaxrange!(MIN_FRICTION, MAX_FRICTION, self.friction, mutation_factor * MUT_FRICTION);
        let radius_range = minmaxrange!(MIN_RADIUS, MAX_RADIUS, self.radius, mutation_factor * MUT_RADIUS);
        let mass_range = minmaxrange!(MIN_MASS, MAX_MASS, self.mass, mutation_factor * MUT_MASS);
        let x_range = minmaxrange!(MIN_X, MAX_X, self.mass, mutation_factor * MUT_XY);
        let y_range = minmaxrange!(MIN_Y, MAX_Y, self.mass, mutation_factor * MUT_XY);
        Node {
            friction: friction_range.ind_sample(&mut rng),
            radius: radius_range.ind_sample(&mut rng),
            mass: mass_range.ind_sample(&mut rng),
            x_gen: x_range.ind_sample(&mut rng),
            y_gen: y_range.ind_sample(&mut rng),
            x: 0.0,
            y: 0.0,
            vx: 0.0,
            vy: 0.0
        }
    }
}

#[test]
fn node_mutate_test() {
    println!("Mutation test");
    let node = Node::new();
    let node2 = node.mutate();
    println!("Mutated node: ");
    println!("Friction: {}", node2.friction);
    println!("Mass: {}", node2.mass);
    println!("Radius: {}", node2.radius);
}

#[derive(Clone)]
struct Muscle {
    force: f64,
    cycle_time1: f64,
    cycle_time2: f64,
    muscle_length1: f64,
    muscle_length2: f64,
    time_in_state: f64,
    state: i64
}

impl Muscle {
    fn new() -> Muscle {
        let mut rng = rand::thread_rng();
        let force_range = Range::new(MIN_FORCE, MAX_FORCE);
        let cycle_range = Range::new(MIN_CYCLE_TIME, MAX_CYCLE_TIME);
        let muscle_range = Range::new(MIN_MUSCLE_LENGTH, MAX_MUSCLE_LENGTH);
        Muscle {
            force : force_range.ind_sample(&mut rng),
            cycle_time1: cycle_range.ind_sample(&mut rng),
            cycle_time2: cycle_range.ind_sample(&mut rng),
            muscle_length1: muscle_range.ind_sample(&mut rng),
            muscle_length2: muscle_range.ind_sample(&mut rng),
            time_in_state: 0.0,
            state: 1
        }
    }

    fn reset(&mut self) {
        self.time_in_state = 0.0;
        self.state = 1;
    }
}

impl Mutate for Muscle {
    fn mutate(&self) -> Muscle {
        let mut rng = rand::thread_rng();
        let distribution = Exp::new(1.0 / MUTATE_LAMBDA);
        let mutation_factor = distribution.ind_sample(&mut rng);
        let force_range = minmaxrange!(MIN_FORCE, MAX_FORCE, self.force, mutation_factor * MUT_FORCE);
        let cycle_range1 = minmaxrange!(MIN_CYCLE_TIME, MAX_CYCLE_TIME, self.cycle_time1, mutation_factor * MUT_CYCLE_TIME);
        let cycle_range2 = minmaxrange!(MIN_CYCLE_TIME, MAX_CYCLE_TIME, self.cycle_time2, mutation_factor * MUT_CYCLE_TIME);
        let muscle_range1 = minmaxrange!(MIN_MUSCLE_LENGTH, MAX_MUSCLE_LENGTH, self.muscle_length1, mutation_factor * MUT_MUSCLE_LENGTH);
        let muscle_range2 = minmaxrange!(MIN_MUSCLE_LENGTH, MAX_MUSCLE_LENGTH, self.muscle_length2, mutation_factor * MUT_MUSCLE_LENGTH);
        Muscle {
            force: force_range.ind_sample(&mut rng),
            cycle_time1: cycle_range1.ind_sample(&mut rng),
            cycle_time2: cycle_range2.ind_sample(&mut rng),
            muscle_length1: muscle_range1.ind_sample(&mut rng),
            muscle_length2: muscle_range2.ind_sample(&mut rng),
            time_in_state: 0.0,
            state: 1
        }
    }
}

// There will always be node_count * (node_count - 1) muscles
#[derive(Clone)]
struct Creature {
    nodes: Vec<Node>,
    muscles: Vec<Muscle>,
    node_count: f64,
    fitness: f64,
    id: i64
}

impl Creature {
    fn new() -> Creature {
        let mut rng = rand::thread_rng();
        let node_range = Range::new(MIN_NODES, MAX_NODES);
        let node_count = node_range.ind_sample(&mut rng);
        let mut nodes: Vec<Node> = Vec::new();
        for _ in 0..(node_count as usize) {
            nodes.push(Node::new());
        }
        let mut muscles: Vec<Muscle> = Vec::new();
        for _ in 0..(((node_count as usize) * (node_count as usize - 1)) / 2) {
            muscles.push(Muscle::new());
        }
        unsafe { next_id += 1; }
        Creature { nodes: nodes, muscles: muscles, node_count: node_count, fitness: 0.0, id: unsafe { next_id }  }
    }

    fn reset(&mut self) {
        self.fitness = 0.0;
        for node in &mut self.nodes {
            node.reset();
        }
        for mut muscle in &mut self.muscles {
            muscle.reset();
        }
    }

    fn get_coord_x(&self) -> f64 {
        self.nodes.iter().fold(0.0, |acc, crt| acc + crt.y) / self.nodes.len() as f64
    }
}

impl Mutate for Creature {
    fn mutate(&self) -> Creature {
        let mut rng = rand::thread_rng();
        let distribution = Exp::new(1.0 / MUTATE_LAMBDA);
        let mutation_factor = distribution.ind_sample(&mut rng);
        let node_range = minmaxrange!(MIN_NODES, MAX_NODES, self.node_count, mutation_factor * MUT_NODES);
        let node_count = node_range.ind_sample(&mut rng);
        let mut nodes: Vec<Node> = self.nodes.mutate();
        let mut muscles: Vec<Muscle> = self.muscles.mutate();
        if node_count as usize > nodes.len() {
            for _ in nodes.len()..(node_count as usize) {
                nodes.push(Node::new());
            }
            for _ in muscles.len()..(((node_count as usize) * (node_count as usize - 1)) / 2) {
                muscles.push(Muscle::new());
            }
        } else if (node_count as usize) < nodes.len() {
            while nodes.len() > node_count as usize {
                nodes.pop();
            }
            while muscles.len() > (node_count as usize) * (node_count as usize - 1) / 2 {
                muscles.pop();
            }
        }
        unsafe { next_id += 1; }
        Creature { nodes: nodes, muscles: muscles, node_count: node_count, fitness: self.fitness, id: unsafe { next_id } }
    }
}

fn step_creature(creature: &mut Creature) {
    // Step the muscles
    let mut muscle_crt = 0;
    for i in 0..creature.nodes.len() {
        for j in (i + 1)..creature.nodes.len() { // muscle is a temporary copy of the muscle
                                           // it only exists for easy of use
            let node1 = creature.nodes[i].clone();
            let node2 = creature.nodes[j].clone();
            let (x1, y1, x2, y2) = (node1.x, node1.y, node2.x, node2.y);
            let mut distance = ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)).sqrt();
            if distance.is_nan() {
                distance = 0.01;
            }
            if distance < 0.01 {
                distance = 0.01;
            }
            //println!("{}", distance);
            let muscle = creature.muscles[muscle_crt].clone();
            let preferred_distance =
                if muscle.state == 1 {
                    muscle.muscle_length1
                } else {
                    muscle.muscle_length2
                };
            let muscle_force = creature.muscles[muscle_crt].force;
            let mut absx = (x2 - x1).abs();
            if absx < 0.01 {
                absx = 0.01;
            }
            let mut absy = (x2 - x1).abs();
            if absy < 0.01 {
                absy = 0.01;
            }
            let muscle_force_x = muscle_force * absx / distance;
            let muscle_force_y = muscle_force * absy / distance;
            if (preferred_distance - distance).abs() > MUSCLE_DELTA_ACCEPT {
                let mut direction = (preferred_distance - distance) / (preferred_distance - distance).abs();
                if direction < 0.01 {
                    direction = 0.01;
                }
                let mut dx = (x2 - x1) / (x2 - x1).abs();
                if dx < 0.01 || dx.is_nan() {
                    dx = 0.01;
                }
                let mut dy = (x2 - x1) / (x2 - x1).abs();
                if dy < 0.01 || dy.is_nan() {
                    dy = 0.01;
                }
                creature.nodes[i].vx += direction * dx * muscle_force_x / node1.mass;
                creature.nodes[i].vy += direction * dy * muscle_force_y / node1.mass;
                creature.nodes[j].vx += direction * -dx * muscle_force_x / node2.mass;
                creature.nodes[j].vy += direction * -dy * muscle_force_y / node2.mass;
            }
            creature.muscles[muscle_crt].time_in_state += STEP_SECONDS;
            if muscle.state == 1 {
                if creature.muscles[muscle_crt].time_in_state > muscle.cycle_time1 {
                    creature.muscles[muscle_crt].time_in_state = 0.0;
                    creature.muscles[muscle_crt].state = 2;
                }
            } else {
                if creature.muscles[muscle_crt].time_in_state > muscle.cycle_time2 {
                    creature.muscles[muscle_crt].time_in_state = 0.0;
                    creature.muscles[muscle_crt].state = 1;
                }
            }
            muscle_crt += 1;
        }
    }
    // And then we step the node positions, accounting for gravity
    for node in &mut creature.nodes {
        //println!("IN {} {} {} {}", node.x as i64, node.y as i64, node.vx as i64, node.vy as i64);
        node.vy += STEP_SECONDS * GRAVITY_ACCEL;
        node.y += node.vy * STEP_SECONDS;
        if node.ground_distance() < 0.0 {
            node.put_on_ground();
            node.vy = 0.0;
        }
        if node.ground_distance() < 0.1 {
            node.vx *= node.friction;
        }
        node.x += node.vx * STEP_SECONDS;
        //println!("{} {}", node.vx * STEP_SECONDS, node.vy * STEP_SECONDS);
        //println!("OUT {} {} {} {}", node.x as i64, node.y as i64, node.vx as i64, node.vy as i64);
    }
}

fn test_fitness(creature: &mut Creature) -> f64 {
    creature.reset();
    for _ in 1..FITNESS_STEPS {
        step_creature(creature);
    }
    creature.get_coord_x()
}

#[test]
fn test_fitness_test() {
    for _ in 1..50000 {
        let creature = Creature::new();
        let fitness = test_fitness(creature);
        for _ in 1..20 {
            let fitness2 = test_fitness(creature);
            assert!(fitness == fitness2);
        }
    }
}

fn test_group_fitness(creatures: &mut Vec<Creature>) {
    for mut creature in creatures {
        creature.reset();
        let fitness = test_fitness(&mut creature);
        creature.fitness = fitness;
    }
}

fn sort_creatures(creatures: &mut Vec<Creature>) {
    creatures.sort_by(|a, b| {
        if a.fitness < b.fitness {
            Ordering::Greater
        } else if a.fitness == b.fitness {
            Ordering::Equal
        } else {
            Ordering::Less
        }
    });
}

fn print_generation_statistics(creatures: &Vec<Creature>) {
    let mut total_fitness = 0.0;
    let mut species_count = [0; 11];
    for creature in creatures {
        total_fitness += creature.fitness;
        species_count[creature.node_count as usize] += 1;
    }
    let average_fitness = total_fitness / creatures.len() as f64;
    let median_fitness = creatures[creatures.len() / 2].fitness;
    let maximum_fitness = creatures[0].fitness;
    println!("Average Fitness: {}", average_fitness);
    println!("Median Fitness: {}", median_fitness);
    println!("Maximum Fitness: {}", maximum_fitness);
    println!("Species: ");
    for i in 3..11 {
        println!("S{}: {} members", i, species_count[i]);
    }
}

fn kill_bottom(creatures: &mut Vec<Creature>) {
    //let max_fitness = creatures[0].fitness;
    //let min_fitness = creatures.last().unwrap().fitness;
    //let mut rng = rand::thread_rng();
    //let distribution = Range::new(min_fitness - 0.1, max_fitness);
    let median_fitness = creatures[creatures.len() / 2].fitness;
    creatures.retain(|c| {
        //let ran_nr = distribution.ind_sample(&mut rng);
        median_fitness <= c.fitness
    });
    println!("Killed creatures down to {}", creatures.len())
}

fn reproduce_to_number(creatures: &mut Vec<Creature>, number: usize) { // Assumes first creature is most fit creature
    let mut rng = rand::thread_rng();
    let distribution = Range::new(0.0, 500.0);//Exp::new(0.5);
    let new_creature = creatures[0].mutate(); // First creature always reproduces twice
    creatures.push(new_creature);
    let new_creature = creatures[0].mutate(); // First creature always reproduces twice
    creatures.push(new_creature);
    let new_creature = creatures[1].mutate(); // Second creature always reproduces once
    creatures.push(new_creature);
    for _ in creatures.len()..(number + 1) {
        let mut creature_number = (distribution.ind_sample(&mut rng) + 1.0) as usize;
        if creature_number > creatures.len() {
            creature_number = creatures.len() - 1;
        }
        let new_creature = creatures[creature_number].mutate();
        creatures.push(new_creature);
    }
}

fn main() {
    let generation_number = 1000;
    let creature_count = 10000;
    let mut creatures : Vec<Creature> = Vec::new();
    for _ in 0..creature_count {
        creatures.push(Creature::new());
    }
    for generation in 1..generation_number {
        println!("Generation #{}: ", generation);
        test_group_fitness(&mut creatures);
        sort_creatures(&mut creatures);
        print_generation_statistics(&creatures);
        kill_bottom(&mut creatures);
        reproduce_to_number(&mut creatures, creature_count);
        println!("\n");
    }
}
