<?php
header("Content-Type: application/json");

$requestBody = file_get_contents('php://input');
$request = json_decode($requestBody, true);

if (isset($request['question'])) {
    $question = $request['question'];
    
    // Run your model here
    // Make sure to use exec() or similar methods to call your Python script
    $command = escapeshellcmd("python3 /path/to/your/model.py \"$question\"");
    $output = shell_exec($command);
    
    echo json_encode(['answer' => $output]);
} else {
    echo json_encode(['answer' => 'Invalid request.']);
}
?>
