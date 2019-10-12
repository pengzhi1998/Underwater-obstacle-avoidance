% Three attenuation parameters which are calculated according to the
% underwater color chart experiment. Those three parameters could be
% adjusted according to the real environment.This method is quite 
% straightforward. Another way is to use waterGan, which is a little 
% bit harder and inefficient.
Red_attenuation = -0.25;
Green_attenuation = -0.060;
Bule_attenuation = -0.025;

underwater_rgb1 = exp(Red_attenuation .* depths(:, :, :)) .* squeeze(double(images(:, :, 1, :)));
underwater_rgb2 = exp(Green_attenuation .* depths(:, :, :)) .* squeeze(double(images(:, :, 2, :)));
underwater_rgb3 = exp(Bule_attenuation .* depths(:, :, :)) .* squeeze(double(images(:, :, 3, :)));
underwater_rgb = cat(4, underwater_rgb1, underwater_rgb2, underwater_rgb3);
underwater_rgb = permute(underwater_rgb, [1, 2, 4, 3]);
underwater_rgb = uint8(underwater_rgb);
save('test.mat', 'underwater_rgb', 'depths', '-v7.3');