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
    A100_48, // 'ampere,a40,48g'
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

fn parse_single_node(input: &str) -> Option<(Node, &str)> {
    println!("Attempting to parse node...");

    let (remaining, node_name) = match parse_node_name(input) {
        Ok(result) => {
            println!("Successfully parsed node name: {}", result.1);
            result
        }
        Err(e) => {
            println!("Failed to parse node name: {:?}", e);
            return None;
        }
    };

    let (remaining, features) = match parse_available_features(remaining) {
        Ok(result) => {
            println!("Successfully parsed features: {}", result.1);
            result
        }
        Err(e) => {
            println!("Failed to parse features: {:?}", e);
            return None;
        }
    };

    let (remaining, cfg_tres) = match parse_cfg_tres(remaining) {
        Ok(result) => {
            println!("Successfully parsed CfgTRES: {}", result.1);
            result
        }
        Err(e) => {
            println!("Failed to parse CfgTRES: {:?}", e);
            return None;
        }
    };

    let (remaining, alloc_tres) = match parse_alloc_tres(remaining) {
        Ok(result) => {
            println!("Successfully parsed AllocTRES: {}", result.1);
            result
        }
        Err(e) => {
            println!("Failed to parse AllocTRES: {:?}", e);
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

    println!("Successfully parsed node: {:?}", node);
    Some((node, remaining))
}

fn parse_nodes(input: &str) -> Vec<Node> {
    let mut nodes = Vec::new();
    let mut current_pos = 0;

    while current_pos < input.len() {
        if let Some((start, end)) = find_next_node(&input[current_pos..]) {
            let node_str = &input[current_pos + start..current_pos + end];
            match parse_single_node(node_str) {
                Some((node, _)) => {
                    nodes.push(node);
                }
                None => {
                    println!("Failed to parse node: {}", node_str);
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
    cpu_available: u32,
    mem_available: u32,
}

fn max_avail_alloc(nodes: &[Node]) -> Vec<(String, GPUType, ResourceAllocation)> {
    use std::collections::HashMap;
    let mut results = Vec::new();

    // For each limiting condition
    for condition in ["GPU", "CPU", "Memory"].iter() {
        // Find max for each GPU type under this condition
        // Create the array without references
        for gpu_type in [
            GPUType::A100_80,
            GPUType::A100_48,
            GPUType::H100,
            GPUType::L40,
            GPUType::A40,
        ]
        .iter()
        .cloned()
        {
            // Use cloned() to get owned values
            // Find the node with max resources for this condition and GPU type
            let max_node = nodes
                .iter()
                .filter(|node| node.gpu_type.as_ref() == Some(&gpu_type))
                .max_by_key(|node| match *condition {
                    "GPU" => node.gpu_available,
                    "CPU" => node.cpu_available,
                    "Memory" => node.mem_available,
                    _ => unreachable!(),
                });

            if let Some(node) = max_node {
                results.push((
                    condition.to_string(),
                    gpu_type, // Now we can use gpu_type directly
                    ResourceAllocation {
                        gpu_available: node.gpu_available,
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
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let filter_gpu_type: Option<GPUType> = args
        .iter()
        .find(|arg| arg.starts_with("-get="))
        .and_then(|arg| {
            let gpu_type_str = arg.strip_prefix("-get=")?;
            match gpu_type_str {
                "A100_80" => Some(GPUType::A100_80),
                "A100_48" => Some(GPUType::A100_48),
                "H100" => Some(GPUType::H100),
                "L40" => Some(GPUType::L40),
                "A40" => Some(GPUType::A40),
                _ => None,
            }
        });

    let mut buffer = String::new();
    match io::stdin().read_to_string(&mut buffer) {
        Ok(_) => {
            let nodes = parse_nodes(&buffer);
            println!("Found {} nodes", nodes.len());

            let maxima = max_avail_alloc(&nodes);
            println!("\nMaximum allocations by limiting condition and GPU type:");

            for (condition, gpu_type, alloc) in maxima {
                if filter_gpu_type.is_none() || filter_gpu_type.as_ref() == Some(&gpu_type) {
                    println!(
                        "\nWhen optimizing for {}, {:?} node maximum allocation:",
                        condition, gpu_type
                    );
                    println!("  GPUs available: {}", alloc.gpu_available);
                    println!("  CPU cores available: {}", alloc.cpu_available);
                    println!("  Memory available (GB): {}", alloc.mem_available);
                }
            }
        }
        Err(e) => {
            eprintln!("Error reading from stdin: {}", e);
        }
    }
}
