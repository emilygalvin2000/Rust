// crates to read in data
use csv::Reader;
use std::fs::File;
use ndarray::{ Array, Array1, Array2 };
use linfa::Dataset;

/// ML crates
use linfa_trees::{DecisionTree, SplitQuality};
use linfa::prelude::*;

//crate to time entire process
use std::time::Instant;

// function to load and return dataset
fn get_dataset() -> Dataset<f32, usize, ndarray::Dim<[usize; 1]>> {
    // intialize reader pointing to data
    let mut reader = Reader::from_path("./src/heart.csv").unwrap();
   
    // extract headers and data from reader
    let headers = get_headers(&mut reader);
    let data = get_data(&mut reader);

    // calculate the index of the target in the header
    let target_index = headers.len() - 1;
    
    // get the features from the headers
    let features = headers[0..target_index].to_vec();

    // retrieve records and targets from data
    let records = get_records(&data, target_index);
    let targets = get_targets(&data, target_index);
   
    // build dataset with records, targets, and features
    return Dataset::new(records, targets)
      .with_feature_names(features);
   }



// function to extract headers from csv file
fn get_headers(reader: &mut Reader<File>) -> Vec<String> {
return reader
    .headers().unwrap().iter()
    .map(|r| r.to_owned())
    .collect();
}

// function to convert feature data into 2d array
fn get_records(data: &Vec<Vec<f32>>, target_index: usize) -> Array2<f32> {
    let mut records: Vec<f32> = vec![];
    for record in data.iter() {
        records.extend_from_slice( &record[0..target_index] );
    }
    return Array::from( records ).into_shape((303, 13)).unwrap();
}

// function to extract target data into 1d array
fn get_targets(data: &Vec<Vec<f32>>, target_index: usize) -> Array1<usize> {
let targets = data
    .iter()
    .map(|record| record[target_index] as usize)
    .collect::<Vec<usize>>();
    return Array::from( targets );
}

// read and parse csv
fn get_data(reader: &mut Reader<File>) -> Vec<Vec<f32>> {
return reader
    .records()
    .map(|r|
    r
        .unwrap().iter()
        .map(|field| field.parse::<f32>().unwrap())
        .collect::<Vec<f32>>()
    )
    .collect::<Vec<Vec<f32>>>();
}

// MAIN FUNCTION

fn main() {
    // timing compile process
    let start = Instant::now();

    // defining dataset
    let dataset = get_dataset();

    // print dataset - uncomment to see entire dataset
    //println!("{:?}", dataset);

    // splitting data into training and testing
    let (train, test) = dataset.split_with_ratio(0.8);

    // create and train model
    let model = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .max_depth(Some(100))
        .min_weight_split(2.0)
        .min_weight_leaf(1.0)
        .fit(&train)
        .expect("Failed to fit model");
    
    // predict on test data
    let predictions = model.predict(&test);
    let actual = test.targets();

    // print predictions and actual targets
    println!("Predictions: {:?}", predictions);
    println!("Actual targets: {:?}", actual);

    // calculating accuracy of model by hand
    let mut correct_count = 0;
    let mut total_count = 0;

    for (&pred, &act) in predictions.iter().zip(actual.iter()) {
        // incrementing for TP and TN
        if pred == act {
            correct_count += 1;
        }
        // incrementing total count
        total_count +=1;
    }

    let accuracy = correct_count as f32 / total_count as f32;
    println!("Accuracy: {:.2}%", accuracy *100.0);
    
    // compute and print confusion matrix - only prints TP of 0 class
    // let cm = predictions.confusion_matrix(&test).expect("Failed to compute confusion matrix");
    // println!("{:?}", cm);

    // end time
    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration)

}