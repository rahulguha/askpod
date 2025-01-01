package main

import (
	// "bufio"
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"glob"

	"github.com/openai/openai-go"
	// "github.com/openai/openai-go/option"

	// "github.com/spf13/viper"
	"github.com/joho/godotenv"
)
import os
import glob
from dotenv import load_dotenv

func main() {
	

	err := godotenv.Load(".env")
	if err != nil{
	log.Fatalf("Error loading .env file: %s", err)
	}

 // Getting and using a value from .env
	OPENAI_API := os.Getenv("OPENAI_API_KEY")

 	fmt.Println(OPENAI_API)
	 client := openai.NewClient()
	 ctx := context.Background()
 
	 file, err := os.Open("NPR3281628569.mp3")
	 if err != nil {
		 panic(err)
	 }
 
	 transcription, err := client.Audio.Transcriptions.New(ctx, openai.AudioTranscriptionNewParams{
		 Model: openai.F(openai.AudioModelWhisper1),
		 File:  openai.F[io.Reader](file),
	 })
	 if err != nil {
		 panic(err)
	 }
	content := transcription.Text
	file, err = os.Create("myfile.txt")  //create a new file
    if err != nil {
        fmt.Println("Error writing file:", err)
        return
    }
	_, err = file.WriteString(content)
   defer file.Close()
	 

	
}

// func getClient() *openai.Client {
// 	// Returns a singleton client. You don't necessarily have to do that
// 	// Given the underlying implementation of otiai10/openaigo, it doesn't really matter
// 	viperenv := viperEnvVariable("OPENAI_API_KEY")
// 	// client := nil
// 	client := openai.NewClient(
// 		option.WithAPIKey(viperenv), // defaults to os.LookupEnv("OPENAI_API_KEY")
// 	)
// 	// client.BaseURL
// 	// if client == nil {
// 	//   client = openai.NewClient(os.Getenv(API_KEY))
// 	//   // needed for Anyscale and OctoAI
// 	//   client.BaseURL = os.Getenv(BASE_URL)
// 	//   // optional but applies to OpenAI
// 	//   client.Organization = os.Getenv(ORG_ID)
// 	// }
// 	return client
// 	// return nil
//   }