import java.io.*;
import java.util.*;

public class Perceptron
{
    // Random number generator for generating data
    Random random = new Random();
    // Useful procedures = nextGaussian() {0..1}
    // nextBoolean() {set false=0, true=1} for the data
    
    //weights for the perceptron
    double[] weights;
    //alpha change value
    double alpha = 0.01;
    //g(in) value
    double gradient = 1;

    // Constructor (initialize the weight vector)
    public Perceptron()
    {
	// Set weights to random values (distributed according to
        // Gaussian with zero mean and unit variance)
        weights = new double[7];
        for (int i = 0; i < weights.length; i++)
        {
            weights[i] = random.nextGaussian();
        }
    }

    // Adds a batch of data to the given vector
    public void addData(int numExamples, Vector examples)
    {
	// Add numExamples examples
	for (int j = 0; j < numExamples; j++) {

	    // An example has bias + 6 inputs + the classification
	    double[] example = new double[8];
	    
	    // The input for the bias is always -1
	    example[0] = -1;

	    // Get other input values
	    for (int i = 1; i < example.length - 1; i++) {
		example[i] = random.nextBoolean() ? 1 : 0;
	    }

	    // Add classification
	    example[example.length - 1] = 
		(example[1] == 1.0 && example[3] == 1.0 && 
		 example[4] == 1.0 && example[6] == 1.0) ? 1: 0;

	    // Add example to vector
	    examples.addElement(example);
	}
    }

    // Gets a classification for the given example
    public double getClassification(double[] example)
    {
	// Compute weighted input to output neuron
        double classificationValue = 0.0;
        
        for (int i = 0; i < weights.length; i++)
        {
            classificationValue += weights[i] * example[i];
        }
	
	// Compute and return predicted classification
	return classificationValue;
    }

    // Train perceptron on data 
    public void trainPerceptron(Vector examples)
    {
        double errorRate = getErrorRate(examples);
	while(getErrorRate(examples) > 0.0)
        {
            //DEBUG System.err.println("Error rate is at " + errorRate + "%");
            
            Iterator exampleList = examples.iterator();
            while(exampleList.hasNext())
            {
                double[] currentExample = (double[])exampleList.next();

                //if the two classifications disagree increment the error count
                if(!classificationsAgree(currentExample))
                {
                    double accuracy = currentExample[currentExample.length - 1] - getClassification(currentExample);
                    
                    //DEBUG System.err.println("Cant solve : " + currentExample[1] + ", " + currentExample[2] + ", " + currentExample[3] + ", " + currentExample[4] + ", " + currentExample[5] + ", " + currentExample[6] + " with classification of " + getClassification(currentExample));
                    
                    for (int i = 0; i < weights.length; i++)
                    {
                        weights[i] = weights[i] + alpha * accuracy * gradient * currentExample[i];
                    }
                }
            }
            
            errorRate = getErrorRate(examples);
        }
    }

    // Computes error rate on given data
    public double getErrorRate(Vector examples)
    {
	// Go through all the examples and count the
	// number of errors returning a fraction that
	// corresponds to error count/# examples
        
        double errorCount = 0.0;
        
        Iterator exampleList = examples.iterator();
        while(exampleList.hasNext())
        {
            double[] currentExample = (double[])exampleList.next();
            
            //if the two classifications disagree increment the error count
            if(!classificationsAgree(currentExample))
            {
                errorCount++;
            }
        }

	return errorCount / examples.size();
    }
    
    //returns true if the perceptron classification matches the actual classification
    private boolean classificationsAgree(double[] example)
    {
        if(getClassification(example) >= 0.5 && example[example.length - 1] == 1)
        {
            return true;
        }
        else if(getClassification(example) < 0.5 && example[example.length - 1] == 0)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    
    // The main control function
    public static void main(String[] args)
    {
	Vector forTraining = new Vector();
	Vector forTesting = new Vector();
	Perceptron perc = new Perceptron();
	
	// Get test data
	perc.addData(500, forTesting);

	// Print the first ten test examples
	System.out.println("\n=======================================");
	System.out.println("10 test examples");
	System.out.println("=======================================");
	for (int i = 0; i < 10; i++) {
	    double[] example = (double[])forTesting.elementAt(i);
	    for (int j = 1; j < example.length; j++) {
		System.out.print(example[j] + " ");
	    }
	    System.out.println();
	}
	System.out.println("=======================================\n");

	// Generate data for learning curve;
	System.out.println("=======================================");
	System.out.println("#Training\tPercent correct");
	System.out.println("=======================================");
	while (forTraining.size() < 500) {
	    perc.addData(50, forTraining);
	    perc.trainPerceptron(forTraining);
	    System.out.println(forTraining.size() + "\t\t" + 
			       (100 * (1 - perc.getErrorRate(forTesting))));
	}
	System.out.println("=======================================\n");
    }
}
