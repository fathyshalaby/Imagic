import torch
from torch.optim import Optimizer as optim


class Imagic:
    def __init__(
        self, diffusion_model, auxiliary_models, text_encoder, target_text, image_size
    ):
        self.diffusion_model = diffusion_model
        self.auxiliary_models = auxiliary_models
        self.text_encoder = text_encoder
        self.target_text = target_text
        self.image_size = image_size
        self.target_embedding = None

    def optimize_embedding(self, num_embedding_steps, learning_rate_embedding):
        self.target_embedding = self.text_encoder.encode(self.target_text).unsqueeze(0)
        self.target_embedding.requires_grad = True

        optimizer_embedding = optim.Adam(
            [self.target_embedding], lr=learning_rate_embedding
        )

        for step in range(num_embedding_steps):
            optimizer_embedding.zero_grad()

            noisy_image = self.diffusion_model.generate_noisy_image(self.image_size)
            generated_image = self.diffusion_model.generate_image_from_embedding(
                noisy_image, self.target_embedding
            )

            loss = torch.mean((generated_image - noisy_image) ** 2)
            loss.backward()

            optimizer_embedding.step()

            if (step + 1) % 100 == 0:
                print(
                    f"Embedding Optimization Step: {step + 1}/{num_embedding_steps}, Loss: {loss.item()}"
                )

        self.target_embedding.requires_grad = False

    def fine_tune_models(self, num_fine_tuning_steps, learning_rate_fine_tuning):
        optimizer_fine_tuning = optim.Adam(
            self.diffusion_model.parameters(), lr=learning_rate_fine_tuning
        )

        for step in range(num_fine_tuning_steps):
            optimizer_fine_tuning.zero_grad()

            noisy_image = self.diffusion_model.generate_noisy_image(self.image_size)
            generated_image = self.diffusion_model.generate_image_from_embedding(
                noisy_image, self.target_embedding
            )

            loss = torch.mean((generated_image - noisy_image) ** 2)
            loss.backward()

            optimizer_fine_tuning.step()

            if (step + 1) % 100 == 0:
                print(
                    f"Fine-Tuning Step: {step + 1}/{num_fine_tuning_steps}, Loss: {loss.item()}"
                )

        # Fine-tuning auxiliary models
        for auxiliary_model in self.auxiliary_models:
            auxiliary_model.train()
            optimizer_auxiliary = optim.Adam(
                auxiliary_model.parameters(), lr=learning_rate_fine_tuning
            )

            for step in range(num_fine_tuning_steps):
                optimizer_auxiliary.zero_grad()

                noisy_image = self.diffusion_model.generate_noisy_image(self.image_size)
                generated_image = self.diffusion_model.generate_image_from_embedding(
                    noisy_image, self.target_embedding
                )
                edited_image = auxiliary_model(generated_image)

                loss = torch.mean((edited_image - noisy_image) ** 2)
                loss.backward()

                optimizer_auxiliary.step()

                if (step + 1) % 100 == 0:
                    print(
                        f"Auxiliary Model Fine-Tuning Step: {step + 1}/{num_fine_tuning_steps}, Loss: {loss.item()}"
                    )

        for auxiliary_model in self.auxiliary_models:
            auxiliary_model.eval()

    def interpolate_embedding(self, eta):
        target_embedding = self.text_encoder.encode(self.target_text).unsqueeze(0)
        interpolated_embedding = (
            eta * target_embedding + (1 - eta) * self.target_embedding
        )
        return interpolated_embedding

    def generate_edited_image(self, interpolated_embedding):
        self.diffusion_model.eval()

        noisy_image = self.diffusion_model.generate_noisy_image(self.image_size)
        edited_image = self.diffusion_model.generate_image_from_embedding(
            noisy_image, interpolated_embedding
        )

        for auxiliary_model in self.auxiliary_models:
            auxiliary_model.eval()
            edited_image = auxiliary_model(edited_image)

        return edited_image
