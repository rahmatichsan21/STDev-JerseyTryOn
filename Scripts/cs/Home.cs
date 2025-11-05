using Godot;
using System;

public partial class Home : Control
{
	// Deklarasi variabel untuk tombol
	private Button startButton;
	private Button aboutButton;
	private Button howToUseButton;
	private Button exitButton;

	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
		// Get node button dari scene tree
		startButton = GetNode<Button>("Start");
		aboutButton = GetNode<Button>("About");
		howToUseButton = GetNode<Button>("How To Use");
		exitButton = GetNode<Button>("Exit");

		// Set text untuk setiap button dari property parent ke child Button
		startButton.GetNode<Button>("Button").Text = startButton.Text;
		aboutButton.GetNode<Button>("Button").Text = aboutButton.Text;
		howToUseButton.GetNode<Button>("Button").Text = howToUseButton.Text;
		exitButton.GetNode<Button>("Button").Text = exitButton.Text;

		// Connect button signals
		startButton.Pressed += OnStartButtonPressed;
		aboutButton.Pressed += OnAboutButtonPressed;
		howToUseButton.Pressed += OnHowToUseButtonPressed;
		exitButton.Pressed += OnExitButtonPressed;
	}

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
	}

	// Handler untuk tombol Start
	private void OnStartButtonPressed()
	{
		GD.Print("Start button pressed");
		// Ganti dengan scene yang sesuai, misalnya main game atau menu selanjutnya
		GetTree().ChangeSceneToFile("res://Scene/Start.tscn");
	}

	// Handler untuk tombol About
	private void OnAboutButtonPressed()
	{
		GD.Print("About button pressed");
		GetTree().ChangeSceneToFile("res://Scene/About.tscn");
	}

	// Handler untuk tombol How To Use
	private void OnHowToUseButtonPressed()
	{
		GD.Print("How To Use button pressed");
		GetTree().ChangeSceneToFile("res://Scene/howToUse.tscn");
	}

	// Handler untuk tombol Exit
	private void OnExitButtonPressed()
	{
		GD.Print("Exit button pressed");
		GetTree().Quit();
	}
}
