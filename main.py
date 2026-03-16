#!/usr/bin/env python3
"""
Mini AI Model Trainer Framework

This module demonstrates advanced Object-Oriented Programming concepts
in Python through a simplified ML training framework simulation.
"""

from abc import ABC, abstractmethod
from typing import List


# =============================================================================
# 1. ModelConfig Class
# Purpose: Stores model configuration settings
# Concepts: Instance attributes, Magic method (__repr__)
# =============================================================================

class ModelConfig:
    """
    Stores model configuration settings.

    This class demonstrates:
    - Instance attributes (model_name, learning_rate, epochs)
    - Magic method (__repr__)
    """

    def __init__(self, model_name: str, learning_rate: float = 0.01, epochs: int = 10):
        # Instance attributes - each instance has its own values
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.epochs = epochs

    def __repr__(self) -> str:
        """
        Magic method for string representation.
        Called when print() is used on the object.
        """
        return f"[Config] {self.model_name} | lr={self.learning_rate} | epochs={self.epochs}"


# =============================================================================
# 2. BaseModel (Abstract Base Class)
# Concepts: Abstraction (ABC), Class attribute, Composition
# =============================================================================

class BaseModel(ABC):
    """
    Abstract base class for all models.

    This class demonstrates:
    - Abstraction: Cannot be instantiated directly, enforces implementing abstract methods
    - Class attribute: model_count shared across all instances
    - Composition: Contains ModelConfig object as part of itself
    """

    # Class attribute - shared across all instances of this class
    model_count = 0

    def __init__(self, config: ModelConfig):
        """
        Constructor demonstrates:
        - Composition: BaseModel 'has a' ModelConfig (owns it)
        """
        # Increment class attribute
        BaseModel.model_count += 1

        # Composition: Storing ModelConfig inside the model
        self.config = config

    @abstractmethod
    def train(self, data: List[float]) -> None:
        """
        Abstract method - must be implemented by subclasses.

        Args:
            data: Training data samples
        """
        pass

    @abstractmethod
    def evaluate(self, data: List[float]) -> float:
        """
        Abstract method - must be implemented by subclasses.

        Args:
            data: Evaluation data samples

        Returns:
            float: Evaluation metric
        """
        pass


# =============================================================================
# 3. LinearRegressionModel
# Concepts: Inheritance, Method overriding, super()
# =============================================================================

class LinearRegressionModel(BaseModel):
    """
    Linear Regression model implementation.

    This class demonstrates:
    - Inheritance: Inherits from BaseModel
    - Method overriding: Overrides train() and evaluate()
    - super(): Calls parent constructor
    """

    def __init__(self, config: ModelConfig):
        """
        Constructor demonstrates super() usage to call parent constructor.

        Args:
            config: ModelConfig object
        """
        # super() calls the parent class constructor
        super().__init__(config)

    def train(self, data: List[float]) -> None:
        """
        Override of abstract train method.

        Demonstrates method overriding - provides specific implementation.
        """
        print(
            f"LinearRegression: Training on {len(data)} samples "
            f"for {self.config.epochs} epochs (lr={self.config.learning_rate})"
        )

    def evaluate(self, data: List[float]) -> float:
        """
        Override of abstract evaluate method.

        Returns simulated MSE (Mean Squared Error).
        """
        # Simulated MSE calculation
        mse = 0.042
        print(f"LinearRegression: Evaluation MSE = {mse}")
        return mse


# =============================================================================
# 4. NeuralNetworkModel
# Concepts: Inheritance, Method overriding, Additional attributes
# =============================================================================

class NeuralNetworkModel(BaseModel):
    """
    Neural Network model implementation.

    This class demonstrates:
    - Inheritance: Inherits from BaseModel
    - Method overriding: Overrides train() and evaluate()
    - Additional instance attributes (layers)
    - super(): Calls parent constructor
    """

    def __init__(self, config: ModelConfig, layers: List[int]):
        """
        Constructor demonstrates:
        - Additional attributes beyond parent class
        - super() to call parent constructor

        Args:
            config: ModelConfig object
            layers: List of integers representing network architecture
        """
        # super() calls the parent class constructor
        super().__init__(config)

        # Additional instance attribute specific to NeuralNetwork
        self.layers = layers

    def train(self, data: List[float]) -> None:
        """
        Override of abstract train method.

        Demonstrates method overriding with additional behavior.
        """
        print(
            f"NeuralNetwork {self.layers}: Training on {len(data)} samples "
            f"for {self.config.epochs} epochs (lr={self.config.learning_rate})"
        )

    def evaluate(self, data: List[float]) -> float:
        """
        Override of abstract evaluate method.

        Returns simulated accuracy.
        """
        # Simulated accuracy calculation
        accuracy = 91.5
        print(f"NeuralNetwork: Evaluation Accuracy = {accuracy}%")
        return accuracy


# =============================================================================
# 5. DataLoader
# Concept: Aggregation (Trainer receives DataLoader but doesn't create it)
# =============================================================================

class DataLoader:
    """
    Simple data loader that stores dataset.

    This class demonstrates:
    - Aggregation concept: Created externally, passed to Trainer
    - The DataLoader exists independently of the Trainer
    """

    def __init__(self, data: List[float]):
        """
        Args:
            data: List of data samples
        """
        self.data = data

    def get_data(self) -> List[float]:
        """Returns the stored data."""
        return self.data


# =============================================================================
# 6. Trainer
# Concepts: Polymorphism, Aggregation
# =============================================================================

class Trainer:
    """
    Training pipeline controller.

    This class demonstrates:
    - Polymorphism: Works with ANY model inheriting from BaseModel
    - Aggregation: Receives DataLoader from outside (doesn't create it)
    """

    def __init__(self, model: BaseModel, data_loader: DataLoader):
        """
        Constructor demonstrates:
        - Aggregation: Trainer receives DataLoader but doesn't create it
        - Polymorphism: model can be any BaseModel subclass

        Args:
            model: Any model inheriting from BaseModel
            data_loader: DataLoader containing the dataset
        """
        # Polymorphism: model can be LinearRegressionModel, NeuralNetworkModel, etc.
        self.model = model
        # Aggregation: DataLoader is passed in from outside
        self.data_loader = data_loader

    def run(self) -> None:
        """
        Executes the training pipeline.

        This method demonstrates polymorphism - it works with any model
        that inherits from BaseModel without knowing the specific type.
        """
        # Get the model name for display
        model_name = self.model.config.model_name

        # Print training header
        print(f"\n--- Training {model_name} ---")

        # Polymorphism in action: calls the appropriate train() method
        # based on the actual type of self.model
        self.model.train(self.data_loader.get_data())

        # Polymorphism in action: calls the appropriate evaluate() method
        # based on the actual type of self.model
        self.model.evaluate(self.data_loader.get_data())


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    """
    Main script demonstrating the Mini AI Model Trainer Framework.

    This demonstrates all OOP concepts working together.
    """

    # 1. Create two configs
    lr_config = ModelConfig("LinearRegression", learning_rate=0.01, epochs=10)
    nn_config = ModelConfig("NeuralNetwork", learning_rate=0.001, epochs=20)

    # 2. Create models
    linear_model = LinearRegressionModel(lr_config)
    neural_model = NeuralNetworkModel(nn_config, layers=[64, 32, 1])

    # 3. Create DataLoader with dataset of 5 samples
    # Aggregation: DataLoader is created externally and passed to Trainer
    data_loader = DataLoader([1, 2, 3, 4, 5])

    # 4. Print configurations
    # Uses __repr__ magic method
    print(lr_config)
    print(nn_config)

    # 5. Print total models created using BaseModel.model_count
    # Accesses class attribute
    print(f"\nModels created: {BaseModel.model_count}")

    # 6. Create Trainer objects and run them
    # Demonstrates polymorphism - same Trainer works with different model types
    trainer_lr = Trainer(linear_model, data_loader)
    trainer_lr.run()

    trainer_nn = Trainer(neural_model, data_loader)
    trainer_nn.run()


if __name__ == "__main__":
    main()