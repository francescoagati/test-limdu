const fs = require('fs');
const serialize = require('serialization');


/*
const limdu = require("limdu");

var TextClassifier = limdu.classifiers.multilabel.BinaryRelevance.bind(0, {
	binaryClassifierType: limdu.classifiers.Winnow.bind(0, {retrain_count: 10})
});

var WordExtractor = function(input, features) {
	input.split(" ").forEach(function(word) {
		features[word]=1;
	});
};


intentClassifier = new limdu.classifiers.EnhancedClassifier({
	classifierType: TextClassifier,  
	normalizer: limdu.features.LowerCaseNormalizer,
	featureExtractor: WordExtractor  
});


intentClassifier.trainBatch([
	{input: "I want an apple", output: "apl"},
	{input: "I want a banana", output: "bnn"},
	{input: "I want chips", output: "cps"},
	]);


function newClassifierFunction() {
	var limdu = require('limdu');
	var TextClassifier = limdu.classifiers.multilabel.BinaryRelevance.bind(0, {
		binaryClassifierType: limdu.classifiers.Winnow.bind(0, {retrain_count: 10})
	});

	var WordExtractor = function(input, features) {
		input.split(" ").forEach(function(word) {
			features[word]=1;
		});
	};
	
	// Initialize a classifier with a feature extractor:
	return new limdu.classifiers.EnhancedClassifier({
		classifierType: TextClassifier,
		featureExtractor: WordExtractor,
		pastTrainingSamples: [], // to enable retraining
	});
}

var intentClassifierString = serialize.toString(intentClassifier, newClassifierFunction);


fs.writeFileSync('./brain.json', intentClassifierString);


console.dir(intentClassifier.classify("I want an apple and a banana"));  // ['apl','bnn']
console.dir(intentClassifier.classify("I WANT AN APPLE AND A BANANA")); 
*/
var brain = fs.readFileSync('./brain.json');
var intentClassifierCopy = serialize.fromString(brain, __dirname);


console.log("Deserialized classifier:");
/*
intentClassifierCopy.classifyAndLog("I want an apple and a banana");  // ['apl','bnn']
intentClassifierCopy.classifyAndLog("I want chips and a doughnut");  // ['cps','dnt']
intentClassifierCopy.trainOnline("I want an elm tree", "elm");
intentClassifierCopy.classifyAndLog("I want doughnut and elm tree");
*/
console.dir(intentClassifierCopy.classify("I want an apple and a banana"));  // ['apl','bnn']
console.dir(intentClassifierCopy.classify("apple   ")); 
