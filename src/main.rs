use nom::{
    IResult,
    Parser, // Add this import
    bytes::complete::take_while1,
    bytes::complete::{tag, take_until},
    character::complete::{digit1, multispace0, multispace1},
    combinator::{map_res, opt},
    error::ErrorKind,
    multi::separated_list0,
    sequence::{delimited, preceded, tuple},
};
use std::io::{self, Read};

#[derive(Debug, PartialEq, Clone)] // Add Clone here
enum GPUType {
    A100_80, // 'ampere,a100,80g'
    A100_48, // 'ampere,a100,48g'
    H100,    // 'hopper,h100,80g'
    L40,     // 'ampere,l40s,48g'
    A40,     // 'ampere,a40,48g'
}

impl GPUType {
    fn from_features(features: &str) -> Option<Self> {
        let features = features.trim_matches('\'');
        match features {
            "ampere,a100,80g" => Some(GPUType::A100_80),
            "ampere,a100,48g" => Some(GPUType::A100_48),
            "hopper,h100,80g" => Some(GPUType::H100),
            "ampere,l40s,48g" => Some(GPUType::L40),
            "ampere,a40,48g" => Some(GPUType::A40),
            _ => None,
        }
    }
    fn memory_size(&self) -> u32 {
        match self {
            GPUType::A100_80 => 80,
            GPUType::A100_48 => 48,
            GPUType::H100 => 80,
            GPUType::L40 => 48,
            GPUType::A40 => 48,
        }
    }

    fn total_gpu_memory(&self, gpu_count: u32) -> u32 {
        self.memory_size() * gpu_count
    }
}

#[derive(Debug)]
struct Node {
    node_name: String,
    gpu_type: Option<GPUType>,
    gpu_count: u32,
    gpu_available: u32,
    cpu_count: u32,
    cpu_available: u32,
    mem_count: u32,     // in GB
    mem_available: u32, // in GB
}

fn parse_memory_value(input: &str) -> Result<u32, std::num::ParseIntError> {
    if input.ends_with('G') {
        input[..input.len() - 1].parse::<u32>()
    } else if input.ends_with('M') {
        // Convert MB to GB, rounding up
        let mb = input[..input.len() - 1].parse::<u32>()?;
        Ok((mb + 1023) / 1024)
    } else {
        input.parse::<u32>()
    }
}

fn parse_tres_field(input: &str, field_prefix: &str) -> Option<u32> {
    input
        .split(',')
        .find(|s| s.starts_with(field_prefix))
        .and_then(|s| s.split('=').nth(1))
        .and_then(|s| s.parse::<u32>().ok())
}

fn parse_tres_memory(input: &str) -> Option<u32> {
    input
        .split(',')
        .find(|s| s.starts_with("mem="))
        .and_then(|s| s.split('=').nth(1))
        .and_then(|s| parse_memory_value(s).ok())
}

fn parse_tres_gpu(input: &str) -> Option<u32> {
    input
        .split(',')
        .find(|s| s.starts_with("gres/gpu="))
        .and_then(|s| s.split('=').nth(1))
        .and_then(|s| s.parse::<u32>().ok())
}

fn parse_node_name(input: &str) -> IResult<&str, &str> {
    preceded(tag("NodeName="), take_while1(|c: char| c != ' ')).parse(input)
}

fn parse_available_features(input: &str) -> IResult<&str, &str> {
    let (input, _) = take_until("AvailableFeatures=").parse(input)?;
    let (input, _) = tag("AvailableFeatures=").parse(input)?;
    take_while1(|c: char| c != '\n').parse(input)
}

fn parse_cfg_tres(input: &str) -> IResult<&str, &str> {
    let (input, _) = take_until("CfgTRES=").parse(input)?;
    let (input, _) = tag("CfgTRES=").parse(input)?;
    take_while1(|c: char| c != '\n').parse(input)
}

fn parse_alloc_tres(input: &str) -> IResult<&str, &str> {
    let (input, _) = take_until("AllocTRES=").parse(input)?;
    let (input, _) = tag("AllocTRES=").parse(input)?;
    // Use alt to handle either empty or non-empty values
    nom::branch::alt((
        take_while1(|c: char| c != '\n'),
        tag(""), // Handle empty case
    ))
    .parse(input)
}

fn find_next_node(input: &str) -> Option<(usize, usize)> {
    let start = input.find("NodeName=")?;
    let end = input[start..]
        .find("\n\nNodeName=")
        .map(|e| e + start)
        .unwrap_or(input.len());
    Some((start, end))
}

fn parse_single_node(input: &str, verbose: bool) -> Option<(Node, &str)> {
    if verbose {
        println!("Attempting to parse node...");
    }

    let (remaining, node_name) = match parse_node_name(input) {
        Ok(result) => {
            if verbose {
                println!("Successfully parsed node name: {}", result.1);
            }
            result
        }
        Err(e) => {
            if verbose {
                println!("Failed to parse node name: {:?}", e);
            }
            return None;
        }
    };

    let (remaining, features) = match parse_available_features(remaining) {
        Ok(result) => {
            if verbose {
                println!("Successfully parsed features: {}", result.1);
            }
            result
        }
        Err(e) => {
            if verbose {
                println!("Failed to parse features: {:?}", e);
            }
            return None;
        }
    };

    let (remaining, cfg_tres) = match parse_cfg_tres(remaining) {
        Ok(result) => {
            if verbose {
                println!("Successfully parsed CfgTRES: {}", result.1);
            }
            result
        }
        Err(e) => {
            if verbose {
                println!("Failed to parse CfgTRES: {:?}", e);
            }
            return None;
        }
    };

    let (remaining, alloc_tres) = match parse_alloc_tres(remaining) {
        Ok(result) => {
            if verbose {
                println!("Successfully parsed AllocTRES: {}", result.1);
            }
            result
        }
        Err(e) => {
            if verbose {
                println!("Failed to parse AllocTRES: {:?}", e);
            }
            return None;
        }
    };

    let gpu_count = parse_tres_gpu(cfg_tres).unwrap_or(0);
    let gpu_alloc = parse_tres_gpu(alloc_tres).unwrap_or(0);
    let cpu_count = parse_tres_field(cfg_tres, "cpu=").unwrap_or(0);
    let cpu_alloc = parse_tres_field(alloc_tres, "cpu=").unwrap_or(0);
    let mem_count = parse_tres_memory(cfg_tres).unwrap_or(0);
    let mem_alloc = parse_tres_memory(alloc_tres).unwrap_or(0);

    let node = Node {
        node_name: node_name.to_string(),
        gpu_type: GPUType::from_features(features),
        gpu_count,
        gpu_available: gpu_count.saturating_sub(gpu_alloc),
        cpu_count,
        cpu_available: cpu_count.saturating_sub(cpu_alloc),
        mem_count,
        mem_available: mem_count.saturating_sub(mem_alloc),
    };

    if verbose {
        println!("Successfully parsed node: {:?}", node);
    }
    Some((node, remaining))
}

fn parse_nodes(input: &str, verbose: bool) -> Vec<Node> {
    let mut nodes = Vec::new();
    let mut current_pos = 0;

    while current_pos < input.len() {
        if let Some((start, end)) = find_next_node(&input[current_pos..]) {
            let node_str = &input[current_pos + start..current_pos + end];
            match parse_single_node(node_str, verbose) {
                Some((node, _)) => {
                    nodes.push(node);
                }
                None => {
                    if verbose {
                        println!("Failed to parse node: {}", node_str);
                    }
                }
            }
            current_pos += end;
        } else {
            break;
        }
    }

    nodes
}

#[derive(Debug)]
struct ResourceAllocation {
    gpu_available: u32,
    gpu_memory: u32, // Total GPU memory in GB
    cpu_available: u32,
    mem_available: u32,
}

#[derive(Debug)]
enum OptimizationTarget {
    GPUType(GPUType), // Optimize for specific GPU type with min 1 GPU
    CPU,              // Optimize for CPU across all nodes
    Memory,           // Optimize for system memory across all nodes
    GPUMem,           // Optimize for GPU memory across all nodes
}

fn max_avail_alloc(
    nodes: &[Node],
    target: &OptimizationTarget,
) -> Vec<(String, Option<GPUType>, ResourceAllocation)> {
    let mut results = Vec::new();

    match target {
        OptimizationTarget::GPUType(gpu_type) => {
            // For each limiting condition
            for condition in ["GPU", "CPU", "Memory"].iter() {
                // Filter nodes of specified type with at least 1 GPU available
                let max_node = nodes
                    .iter()
                    .filter(|node| {
                        node.gpu_type.as_ref() == Some(gpu_type) && node.gpu_available > 0
                    })
                    .max_by_key(|node| match *condition {
                        "GPU" => node.gpu_available,
                        "CPU" => node.cpu_available,
                        "Memory" => node.mem_available,
                        _ => unreachable!(),
                    });

                if let Some(node) = max_node {
                    results.push((
                        condition.to_string(),
                        Some(gpu_type.clone()),
                        ResourceAllocation {
                            gpu_available: node.gpu_available,
                            gpu_memory: gpu_type.total_gpu_memory(node.gpu_available),
                            cpu_available: node.cpu_available,
                            mem_available: node.mem_available,
                        },
                    ));
                }
            }
        }
        OptimizationTarget::CPU => {
            // Find node with maximum available CPUs
            if let Some(node) = nodes.iter().max_by_key(|node| node.cpu_available) {
                let gpu_memory = node
                    .gpu_type
                    .as_ref()
                    .map(|gt| gt.total_gpu_memory(node.gpu_available))
                    .unwrap_or(0);

                results.push((
                    "CPU".to_string(),
                    node.gpu_type.clone(),
                    ResourceAllocation {
                        gpu_available: node.gpu_available,
                        gpu_memory,
                        cpu_available: node.cpu_available,
                        mem_available: node.mem_available,
                    },
                ));
            }
        }
        OptimizationTarget::Memory => {
            // Find node with maximum available memory
            if let Some(node) = nodes.iter().max_by_key(|node| node.mem_available) {
                let gpu_memory = node
                    .gpu_type
                    .as_ref()
                    .map(|gt| gt.total_gpu_memory(node.gpu_available))
                    .unwrap_or(0);

                results.push((
                    "Memory".to_string(),
                    node.gpu_type.clone(),
                    ResourceAllocation {
                        gpu_available: node.gpu_available,
                        gpu_memory,
                        cpu_available: node.cpu_available,
                        mem_available: node.mem_available,
                    },
                ));
            }
        }
        OptimizationTarget::GPUMem => {
            // Find node with maximum total GPU memory
            if let Some(node) = nodes
                .iter()
                .filter(|node| node.gpu_type.is_some())
                .max_by_key(|node| {
                    node.gpu_type
                        .as_ref()
                        .map(|gt| gt.total_gpu_memory(node.gpu_available))
                        .unwrap_or(0)
                })
            {
                let gpu_memory = node
                    .gpu_type
                    .as_ref()
                    .map(|gt| gt.total_gpu_memory(node.gpu_available))
                    .unwrap_or(0);

                results.push((
                    "GPUMem".to_string(),
                    node.gpu_type.clone(),
                    ResourceAllocation {
                        gpu_available: node.gpu_available,
                        gpu_memory,
                        cpu_available: node.cpu_available,
                        mem_available: node.mem_available,
                    },
                ));
            }
        }
    }

    results
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let optimization_target: Option<OptimizationTarget> = args
        .iter()
        .find(|arg| arg.starts_with("-get="))
        .and_then(|arg| {
            let target_str = arg.strip_prefix("-get=")?;
            match target_str {
                "A100_80" => Some(OptimizationTarget::GPUType(GPUType::A100_80)),
                "A100_48" => Some(OptimizationTarget::GPUType(GPUType::A100_48)),
                "H100" => Some(OptimizationTarget::GPUType(GPUType::H100)),
                "L40" => Some(OptimizationTarget::GPUType(GPUType::L40)),
                "A40" => Some(OptimizationTarget::GPUType(GPUType::A40)),
                "CPU" => Some(OptimizationTarget::CPU),
                "Memory" => Some(OptimizationTarget::Memory),
                "GPUMem" => Some(OptimizationTarget::GPUMem),
                _ => None,
            }
        });

    // Only print parsing debug messages if no command line flag is passed.
    let verbose = optimization_target.is_none();

    let mut buffer = String::new();
    match io::stdin().read_to_string(&mut buffer) {
        Ok(_) => {
            let nodes = parse_nodes(&buffer, verbose);
            println!("Found {} nodes", nodes.len());

            match optimization_target {
                Some(target) => {
                    let maxima = max_avail_alloc(&nodes, &target);
                    for (condition, gpu_type, alloc) in maxima {
                        println!(
                            "\nWhen optimizing for {}, maximum allocation{}: ",
                            condition,
                            gpu_type.map_or("".to_string(), |t| format!(" on {:?}", t))
                        );
                        println!("  GPUs available: {}", alloc.gpu_available);
                        println!("  GPU Memory available (GB): {}", alloc.gpu_memory);
                        println!("  CPU cores available: {}", alloc.cpu_available);
                        println!("  System Memory available (GB): {}", alloc.mem_available);
                    }
                }
                None => {
                    // Show all GPU types
                    for gpu_type in [
                        GPUType::A100_80,
                        GPUType::A100_48,
                        GPUType::H100,
                        GPUType::L40,
                        GPUType::A40,
                    ]
                    .iter()
                    {
                        let maxima =
                            max_avail_alloc(&nodes, &OptimizationTarget::GPUType(gpu_type.clone()));
                        for (condition, _, alloc) in maxima {
                            println!(
                                "\nWhen optimizing for {} on {:?} node maximum allocation:",
                                condition, gpu_type
                            );
                            println!("  GPUs available: {}", alloc.gpu_available);
                            println!("  CPU cores available: {}", alloc.cpu_available);
                            println!("  Memory available (GB): {}", alloc.mem_available);
                        }
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Error reading from stdin: {}", e);
        }
    }
}
