underwater_rgb1 = exp(-0.25 .* depths(:, :, :)) .* squeeze(double(images(:, :, 1, :)));
underwater_rgb2 = exp(-0.060 .* depths(:, :, :)) .* squeeze(double(images(:, :, 2, :)));
underwater_rgb3 = exp(-0.025 .* depths(:, :, :)) .* squeeze(double(images(:, :, 3, :)));
underwater_rgb = cat(4, underwater_rgb1, underwater_rgb2, underwater_rgb3);
underwater_rgb = permute(underwater_rgb, [1, 2, 4, 3]);
underwater_rgb = uint8(underwater_rgb);
save('test.mat', 'underwater_rgb', 'depths', '-v7.3');