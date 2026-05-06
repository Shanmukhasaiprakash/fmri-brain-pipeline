%  CONFIG - paths and parameters

NIFTI_PATH = 'C:\Users\shanm\OneDrive\Desktop\Image Processing and Visualization\8_finger_foot_lips.nii';
TR_SECONDS = 2.5;          % ds000114 finger_foot_lips TR
HP_CUTOFF  = 128;          % high-pass filter cutoff in seconds
FWHM_MM    = 6;            % spatial smoothing FWHM
OUT_DIR    = 'C:\Users\shanm\OneDrive\Desktop\Image Processing and Visualization\results';

if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end

%  STEP 1 - LOAD NIfTI DATA

fprintf('[1/7] Loading NIfTI data...\n');

if ~exist(NIFTI_PATH, 'file')
    error(['File not found: ' NIFTI_PATH ...
           '\nPlease check the path is correct.']);
end

fid = fopen(NIFTI_PATH, 'rb', 'l');

% Read dimensions from NIfTI-1 header
fseek(fid, 40, 'bof');
dim_raw = fread(fid, 8, 'int16');
nx = dim_raw(2);
ny = dim_raw(3);
nz = dim_raw(4);
nt = max(dim_raw(5), 1);

% Read datatype
fseek(fid, 108, 'bof');
datatype = fread(fid, 1, 'int16');

% Jump to data (vox_offset = 352 for .nii single-file format)
fseek(fid, 352, 'bof');

switch datatype
    case 4,   prec = 'int16';
    case 16,  prec = 'float32';
    case 64,  prec = 'float64';
    otherwise, prec = 'int16';
end

raw    = fread(fid, nx*ny*nz*nt, prec);
fclose(fid);

data4D = double(reshape(raw, nx, ny, nz, nt));
TR     = TR_SECONDS;

fprintf('    Loaded: [%d x %d x %d x %d]  TR=%.2fs\n', nx, ny, nz, nt, TR);
fprintf('    Scan duration: %.1f min\n\n', nt*TR/60);

% Brain mask: voxels with mean signal above 30% of overall mean
brain_mask = mean(data4D, 4) > mean(data4D(:)) * 0.3;

%  STEP 2 - PREPROCESSING

fprintf('[2/7] Preprocessing...\n');

% --- 2a. Motion correction (phase-correlation translation estimate) -------
motion_params = zeros(nt, 6);
ref_vol = mean(data4D(:,:,:,1:min(10,nt)), 4);
for t = 1:nt
    curr   = data4D(:,:,:,t);
    ref_sl = ref_vol(:,:,round(nz/2));
    mov_sl = curr(:,:,round(nz/2));
    R = abs(fftshift(ifft2(fft2(ref_sl) .* conj(fft2(mov_sl)))));
    [~, idx] = max(R(:));
    [iy, ix] = ind2sub(size(R), idx);
    tx = iy - (size(R,1)/2+1);
    ty = ix - (size(R,2)/2+1);
    motion_params(t,:) = [tx, ty, 0, 0, 0, 0];
    data4D(:,:,:,t) = circshift(curr, [-round(tx), -round(ty), 0]);
end
FD = [0; sum(abs(diff(motion_params(:,1:3))), 2)];
fprintf('    Motion correction done. Mean FD: %.3f mm\n', mean(FD));

% --- 2b. Temporal high-pass filter (FFT-based) ---------------------------
cutoff_idx = max(1, round((1/HP_CUTOFF) / (1/(2*TR)) * (nt/2)));
mean_vol   = mean(data4D, 4);
for z = 1:nz
    sl = reshape(data4D(:,:,z,:), nx*ny, nt);
    F  = fft(sl, [], 2);
    F(:, 1:cutoff_idx) = 0;
    if cutoff_idx > 1
        F(:, end-cutoff_idx+2:end) = 0;
    end
    data4D(:,:,z,:) = reshape(real(ifft(F, [], 2)), nx, ny, nt);
end
data4D = data4D + mean_vol;
fprintf('    High-pass filter applied (%ds cutoff)\n', HP_CUTOFF);

% --- 2c. Spatial Gaussian smoothing --------------------------------------
sigma_vox = (FWHM_MM/2) / 2.355;
r_k = ceil(3*sigma_vox);
[kx, ky, kz] = ndgrid(-r_k:r_k, -r_k:r_k, -r_k:r_k);
kernel = exp(-(kx.^2 + ky.^2 + kz.^2) / (2*sigma_vox^2));
kernel = kernel / sum(kernel(:));
for t = 1:nt
    data4D(:,:,:,t) = convn(data4D(:,:,:,t), kernel, 'same');
end
fprintf('    Spatial smoothing done (FWHM=%dmm)\n\n', FWHM_MM);

%  STEP 3 - FEATURE EXTRACTION (tSNR + GLM)

fprintf('[3/7] Feature extraction...\n');

% tSNR map
mu_vol  = mean(data4D, 4);
sig_vol = std(data4D, 0, 4);
sig_vol(sig_vol == 0) = eps;
tSNR_map = mu_vol ./ sig_vol;
tSNR_map(tSNR_map < 0) = 0;
fprintf('    Mean tSNR: %.1f | Max: %.1f\n', ...
        mean(tSNR_map(brain_mask)), max(tSNR_map(:)));

% GLM - General Linear Model
% EXACT timings from task-fingerfootlips_events.tsv (ds000114):
% 15 blocks of 15s each (Finger/Foot/Lips alternating), one every 30s
% onset:    10,40,70,100,130,160,190,220,250,280,310,340,370,400,430
% duration: 15s each, inter-block rest = 15s
% All three movement types modeled together (any motor > rest)
t_vec    = (0:nt-1)' * TR;

task_onsets_s   = [10,40,70,100,130,160,190,220,250,280,310,340,370,400,430];
task_duration_s = 15;

task_box = zeros(nt, 1);
for b = 1:length(task_onsets_s)
    onset_vol = round(task_onsets_s(b) / TR) + 1;
    end_vol   = min(round((task_onsets_s(b) + task_duration_s) / TR), nt);
    if onset_vol <= nt
        task_box(onset_vol:min(end_vol,nt)) = 1;
    end
end
fprintf('    Exact TSV timings: %d x 15s blocks (Finger/Foot/Lips)\n', ...
        length(task_onsets_s));

t_hrf = (0:TR:30)';
a1=6; b1=1; c2=0.35; a2=16; b2=1;
hrf = (t_hrf.^(a1-1).*exp(-t_hrf/b1) / (factorial(a1-1)*b1^a1)) ...
    - c2*(t_hrf.^(a2-1).*exp(-t_hrf/b2) / (factorial(a2-1)*b2^a2));
hrf(t_hrf < 0) = 0;
hrf = hrf / max(hrf);

% Try to read actual event timings from TSV file if present
events_path = strrep(NIFTI_PATH, '.nii', '_events.tsv');
events_path2 = fullfile(fileparts(NIFTI_PATH), 'task-fingerfootlips_events.tsv');
if exist(events_path, 'file')
    fprintf('    Reading events from: %s\n', events_path);
    events = readtable(events_path, 'FileType','text', 'Delimiter','\t');
    task_box = zeros(nt, 1);
    for e = 1:height(events)
        onset_tr  = round(events.onset(e) / TR) + 1;
        dur_tr    = round(events.duration(e) / TR);
        idx_range = onset_tr : min(onset_tr+dur_tr-1, nt);
        task_box(idx_range) = 1;
    end
    fprintf('    Loaded %d events from TSV\n', height(events));
elseif exist(events_path2, 'file')
    fprintf('    Reading events from: %s\n', events_path2);
    events = readtable(events_path2, 'FileType','text', 'Delimiter','\t');
    task_box = zeros(nt, 1);
    for e = 1:height(events)
        onset_tr  = round(events.onset(e) / TR) + 1;
        dur_tr    = round(events.duration(e) / TR);
        idx_range = onset_tr : min(onset_tr+dur_tr-1, nt);
        task_box(idx_range) = 1;
    end
    fprintf('    Loaded %d events from TSV\n', height(events));
else
    fprintf('    Built-in timing used (TSV not found in folder)\n');
end

task_reg = conv(task_box, hrf, 'full');
task_reg = task_reg(1:nt);
task_reg = (task_reg - mean(task_reg)) / (std(task_reg) + eps);

% Design matrix: [task | intercept | linear drift]
X  = [task_reg, ones(nt,1), linspace(-1,1,nt)'];
p  = size(X, 2);
df = nt - p;

% Use pseudoinverse - robust against near-singular matrices (real data)
pX       = pinv(X);
c_con    = [1; 0; 0];
c_XtX_c  = c_con' * pinv(X'*X) * c_con;

Y_mat    = reshape(data4D, nx*ny*nz, nt)';
beta_mat = pX * Y_mat;
Y_hat    = X * beta_mat;
resid    = Y_mat - Y_hat;
MSE      = sum(resid.^2, 1) / max(df, 1);
effect   = c_con' * beta_mat;
se       = sqrt(c_XtX_c * MSE);
se(se == 0) = eps;

t_map = reshape(effect ./ se, nx, ny, nz);
t_map(~brain_mask) = 0;
t_map(isnan(t_map) | isinf(t_map)) = 0;

fprintf('    GLM done. Peak t=%.2f | Sig voxels (p<0.001): %d\n\n', ...
        max(t_map(:)), sum(t_map(:) > 2.5));

%  STEP 4 - ROI IDENTIFICATION

fprintf('[4/7] Identifying ROIs...\n');

t_thresh = 2.5;   % p<0.01 — appropriate for real data with tSNR~35
k_min    = 10;    % smaller minimum cluster for real data
CC       = bwconncomp(t_map > t_thresh, 6);

roi_mask  = zeros(nx, ny, nz);
roi_count = 0;
roi_stats = struct('size',{}, 'peak_t',{}, 'centroid',{}, 'mean_tSNR',{});

for ci = 1:CC.NumObjects
    v = CC.PixelIdxList{ci};
    if length(v) < k_min, continue; end
    roi_count = roi_count + 1;
    roi_mask(v) = roi_count;
    roi_stats(roi_count).size      = length(v);
    roi_stats(roi_count).peak_t    = max(t_map(v));
    roi_stats(roi_count).mean_tSNR = mean(tSNR_map(v));
    [ix,iy,iz] = ind2sub([nx,ny,nz], v);
    roi_stats(roi_count).centroid  = [mean(ix), mean(iy), mean(iz)];
end

cx = nx/2; cy = ny/2; cz = nz/2;
roi_labels = cell(roi_count, 1);
for r = 1:roi_count
    cen = roi_stats(r).centroid;
    if     cen(3) > cz*1.3 && cen(1) > cx, roi_labels{r} = 'R Motor Cortex';
    elseif cen(3) > cz*1.3,                 roi_labels{r} = 'L Motor Cortex';
    elseif cen(2) > cy*1.4,                 roi_labels{r} = 'Prefrontal Cortex';
    elseif cen(3) < cz*0.6,                 roi_labels{r} = 'Cerebellum';
    elseif cen(1) < cx*0.8,                 roi_labels{r} = 'L Sensory Cortex';
    else,                                   roi_labels{r} = sprintf('Cluster_%d',r);
    end
    fprintf('    ROI %d: %-20s | voxels:%4d | peak-t:%.2f\n', ...
            r, roi_labels{r}, roi_stats(r).size, roi_stats(r).peak_t);
end
if roi_count == 0
    fprintf('    No ROIs above threshold. Check GLM output.\n');
end
fprintf('\n');

%  STEP 5 - FUNCTIONAL CONNECTIVITY

fprintf('[5/7] Functional connectivity...\n');

n_rois    = roi_count;
FC_matrix = [];
roi_ts    = [];

if n_rois > 1
    vox_data = reshape(data4D, nx*ny*nz, nt);
    roi_ts   = zeros(n_rois, nt);
    for r = 1:n_rois
        m = roi_mask == r;
        roi_ts(r,:) = mean(vox_data(m(:),:), 1);
    end
    roi_ts_z = (roi_ts - mean(roi_ts,2)) ./ (std(roi_ts,0,2) + eps);
    r_mat    = real((roi_ts_z * roi_ts_z') / (nt-1));
    r_mat    = max(-0.9999, min(0.9999, r_mat));
    r_mat(1:n_rois+1:end) = 0;
    FC_matrix = 0.5 * log((1 + r_mat) ./ (1 - r_mat));
    FC_matrix = real(FC_matrix);
    FC_matrix(isinf(FC_matrix) | isnan(FC_matrix)) = 0;
    fprintf('    Connectivity matrix: %dx%d ROIs\n\n', n_rois, n_rois);
else
    fprintf('    Skipping (need at least 2 ROIs)\n\n');
end

%  STEP 6 - IMAGE ENHANCEMENT
%  Histogram EQ | Edge Detection | Morphological Operations

fprintf('[6/7] Image enhancement analysis...\n');

% Pick the slice with the most activation for demonstration
t_map_abs = abs(t_map);
slice_scores = squeeze(sum(sum(t_map_abs, 1), 2));
[~, best_z]  = max(slice_scores);
if best_z < 1, best_z = round(nz/2); end

ref_slice = mu_vol(:,:,best_z);
sl_norm   = ref_slice - min(ref_slice(:));
sl_norm   = sl_norm / (max(sl_norm(:)) + eps);

% A. Histogram equalization
sl_histeq = histeq(sl_norm);

% B. Edge detection
sl_canny = edge(sl_norm, 'Canny', [0.05 0.15]);
sl_sobel = edge(sl_norm, 'Sobel');

% C. Morphological operations on ROI binary mask
roi_sl_bin  = roi_mask(:,:,best_z) > 0;
se_disk     = strel('disk', 2);
roi_eroded  = imerode(roi_sl_bin, se_disk);
roi_dilated = imdilate(roi_sl_bin, se_disk);
roi_outline = roi_dilated & ~roi_eroded;
roi_opened  = imopen(roi_sl_bin, strel('disk',1));

% Build 12-panel enhancement figure
fig_enh = figure('Name','Enhancement','Position',[50 50 1400 900],'Color','k');

enh_panels = {sl_norm,'Original slice','gray'; ...
              sl_histeq,'Histogram equalization','gray'; ...
              sl_canny,'Edge detection (Canny)','gray'; ...
              sl_sobel,'Edge detection (Sobel)','gray'};
for p = 1:4
    ax = subplot(3,4,p);
    imagesc(ax, enh_panels{p,1}'); axis(ax,'image','off');
    colormap(ax, enh_panels{p,3});
    title(ax, enh_panels{p,2},'Color','w','FontSize',10,'FontWeight','bold');
    set(ax,'Color','k');
end

morph_panels = {double(roi_sl_bin),'ROI mask (binary)'; ...
                double(roi_eroded),'After erosion (shrink)'; ...
                double(roi_dilated),'After dilation (expand)'; ...
                double(roi_outline),'Morphological outline'; ...
                double(roi_opened),'After opening (denoise)'};
for p = 1:5
    ax = subplot(3,4,4+p);
    imagesc(ax, morph_panels{p,1}'); axis(ax,'image','off');
    colormap(ax,'hot');
    title(ax, morph_panels{p,2},'Color','w','FontSize',10,'FontWeight','bold');
    set(ax,'Color','k');
end

% Histogram comparison
ax_h = subplot(3,4,10);
hold(ax_h,'on');
histogram(ax_h, sl_norm(:), 40,'FaceColor',[0.5 0.5 0.5],'EdgeColor','none','FaceAlpha',0.8);
histogram(ax_h, sl_histeq(:),40,'FaceColor',[1 0.5 0.1],'EdgeColor','none','FaceAlpha',0.8);
xlabel(ax_h,'Intensity','Color','w');
ylabel(ax_h,'Count','Color','w');
title(ax_h,'Histogram: before vs after EQ','Color','w','FontSize',10,'FontWeight','bold');
legend(ax_h,{'Original','Equalized'},'TextColor','w','EdgeColor','none','Location','northeast');
set(ax_h,'Color','k','XColor','w','YColor','w');

% Final overlay: anatomy + activation + morphological outline
ax_ov = subplot(3,4,[11 12]);
imshow(sl_norm', [0 1], 'Parent', ax_ov);
colormap(ax_ov, gray);
hold(ax_ov, 'on');

t_sl_ov  = t_map(:,:,best_z)';
t_sl_ov(t_sl_ov < 3.1) = 0;
t_max_ov = max(t_map(:));

if t_max_ov > 3.1
    cmap_hot = hot(256);
    t_n = t_sl_ov / t_max_ov;
    [nr2, nc2] = size(t_sl_ov);
    ov_rgb = zeros(nr2, nc2, 3);
    for ch = 1:3
        ov_rgb(:,:,ch) = reshape(cmap_hot(max(1,min(256,round(t_n(:)*255+1))),ch), nr2, nc2);
    end
    h_act = imshow(ov_rgb, 'Parent', ax_ov);
    h_act.AlphaData = double(t_sl_ov > 0) * 0.75;
end

B_ov = bwboundaries(roi_outline);
for b = 1:length(B_ov)
    plot(ax_ov, B_ov{b}(:,2), B_ov{b}(:,1), 'c-', 'LineWidth', 1.5);
end
title(ax_ov, 'Final: anatomy + GLM activation + morphological outline', ...
      'Color','w','FontSize',10,'FontWeight','bold');
set(ax_ov,'Color','k');

sgtitle(sprintf('Image Enhancement Analysis - Slice z=%d | BINF 7550 | Jeelakarra & Kakumanu', best_z), ...
        'Color','w','FontSize',12,'FontWeight','bold');
saveas(fig_enh, fullfile(OUT_DIR,'02_image_enhancement.png'));
fprintf('    Saved: 02_image_enhancement.png\n\n');

%  STEP 7 - VISUALIZATION FIGURES

fprintf('[7/7] Generating visualization figures...\n');

% Shared variables for all figures
mean_anat = mu_vol / (max(mu_vol(:)) + eps);
t_ov      = t_map;
t_ov(t_ov < 3.1) = 0;
t_max_val = max(t_map(:));
if t_max_val <= 0, t_max_val = 1; end

% ---- Figure 1: Preprocessing QC ----------------------------------------
fig1 = figure('Name','QC','Position',[50 50 1400 850],'Color','k');
t_ax = (0:nt-1)*TR;

subplot(2,3,1);
plot(t_ax, motion_params(:,1), 'r', 'LineWidth',1.5); hold on;
plot(t_ax, motion_params(:,2), 'g', 'LineWidth',1.5);
xlabel('Time (s)','Color','w'); ylabel('mm','Color','w');
title('Head translations','Color','w','FontWeight','bold');
legend({'X','Y'},'TextColor','w','EdgeColor','w');
set(gca,'Color','k','XColor','w','YColor','w');

subplot(2,3,2);
area(t_ax, FD,'FaceColor',[0.3 0.6 1],'EdgeColor','none');
xlabel('Time (s)','Color','w'); ylabel('FD (mm)','Color','w');
title('Framewise displacement','Color','w','FontWeight','bold');
set(gca,'Color','k','XColor','w','YColor','w');

subplot(2,3,3);
tSNR_brain = tSNR_map(tSNR_map > 5);
histogram(tSNR_brain, 50,'FaceColor',[1 0.5 0],'EdgeColor','none');
xline(mean(tSNR_brain),'--c','LineWidth',2);
xlabel('tSNR','Color','w'); ylabel('Voxels','Color','w');
title(sprintf('tSNR distribution (mean=%.0f)', mean(tSNR_brain)),'Color','w','FontWeight','bold');
set(gca,'Color','k','XColor','w','YColor','w');

sl_idx3   = [round(nz*0.3), round(nz*0.5), round(nz*0.7)];
sl_titles = {'Inferior','Middle','Superior'};
for si = 1:3
    subplot(2,3,3+si);
    imagesc(tSNR_map(:,:,sl_idx3(si))'); axis image off;
    colormap(gca,'hot');
    cb = colorbar; cb.Color = 'w';
    title(sl_titles{si},'Color','w','FontWeight','bold');
end
sgtitle('Preprocessing QC - NeuREI Lab | BINF 7550','Color','w','FontSize',13,'FontWeight','bold');
saveas(fig1, fullfile(OUT_DIR,'01_preprocessing_qc.png'));
fprintf('    Saved: 01_preprocessing_qc.png\n');

% ---- Figure 2: Activation Maps ------------------------------------------
fig2 = figure('Name','Activation','Position',[50 50 1400 500],'Color','k');

% Find the slice with peak activation in each dimension
[~, peak_vox] = max(t_map(:));
[pv_x, pv_y, pv_z] = ind2sub([nx,ny,nz], peak_vox);
fprintf('    Peak activation at voxel [%d %d %d]\n', pv_x, pv_y, pv_z);

view_defs = {'Axial',   pv_z, 3; ...
             'Coronal', pv_y, 2; ...
             'Sagittal',pv_x, 1};

for v = 1:3
    ax = subplot(1,3,v);
    dim    = view_defs{v,3};
    sl_idx = view_defs{v,2};
    switch dim
        case 3
            anat_sl = mean_anat(:,:,sl_idx)';
            t_sl    = t_ov(:,:,sl_idx)';
            roi_sl  = (roi_mask(:,:,sl_idx) > 0)';
        case 2
            anat_sl = squeeze(mean_anat(:,sl_idx,:))';
            t_sl    = squeeze(t_ov(:,sl_idx,:))';
            roi_sl  = (squeeze(roi_mask(:,sl_idx,:)) > 0)';
        case 1
            anat_sl = squeeze(mean_anat(sl_idx,:,:))';
            t_sl    = squeeze(t_ov(sl_idx,:,:))';
            roi_sl  = (squeeze(roi_mask(sl_idx,:,:)) > 0)';
    end

    imshow(anat_sl, [0 1], 'Parent', ax);
    colormap(ax, gray(256));
    hold(ax, 'on');

    % Activation overlay
    cmap_h  = hot(256);
    t_norm  = t_sl / t_max_val;
    t_norm(t_sl == 0) = 0;
    act_mask = t_sl > 0;
    [nr, nc] = size(anat_sl);
    ov = zeros(nr, nc, 3);
    for ch = 1:3
        ov(:,:,ch) = reshape(cmap_h(max(1,min(256,round(t_norm(:)*255+1))),ch), nr, nc);
    end
    h_ov = imshow(ov, 'Parent', ax);
    h_ov.AlphaData = double(act_mask) * 0.78;

    % ROI boundaries
    B = bwboundaries(roi_sl);
    for b = 1:length(B)
        plot(ax, B{b}(:,2), B{b}(:,1), 'c-', 'LineWidth', 1.8);
    end

    title(ax, view_defs{v,1},'Color','w','FontSize',13,'FontWeight','bold');
    colormap(ax, hot);
    cb = colorbar(ax); cb.Color = 'w';
    cb.Label.String = 't-statistic'; cb.Label.Color = 'w';
    if t_max_val > 3.1
        clim(ax, [3.1 t_max_val]);
    end
end
sgtitle('Activation Maps (GLM, p<0.001) - NeuREI Lab | BINF 7550', ...
        'Color','w','FontSize',13,'FontWeight','bold');
saveas(fig2, fullfile(OUT_DIR,'03_activation_maps.png'));
fprintf('    Saved: 03_activation_maps.png\n');

% ---- Figure 3: Functional Connectivity ----------------------------------
if ~isempty(FC_matrix) && n_rois > 1
    fig3 = figure('Name','Connectivity','Position',[100 100 900 700],'Color','k');

    subplot(1,2,1);
    imagesc(FC_matrix);
    colormap(gca,'jet'); axis square;
    cb = colorbar; cb.Color = 'w';
    cb.Label.String = 'Fisher z(r)'; cb.Label.Color = 'w';
    set(gca,'XTick',1:n_rois,'YTick',1:n_rois, ...
            'XTickLabel',roi_labels,'YTickLabel',roi_labels, ...
            'XTickLabelRotation',35,'FontSize',8, ...
            'Color','k','XColor','w','YColor','w');
    title('FC matrix','Color','w','FontWeight','bold');
    for i = 1:n_rois
        for j = 1:n_rois
            if abs(FC_matrix(i,j)) > 0.3
                tc = 'k';
            else
                tc = 'w';
            end
            text(j, i, sprintf('%.2f', FC_matrix(i,j)), ...
                 'HorizontalAlignment','center','FontSize',7,'Color',tc);
        end
    end

    subplot(1,2,2);
    theta = linspace(0, 2*pi, n_rois+1); theta(end) = [];
    nx2 = cos(theta); ny2 = sin(theta);
    hold on;
    for i = 1:n_rois
        for j = i+1:n_rois
            if abs(FC_matrix(i,j)) > 0.15
                w   = min(abs(FC_matrix(i,j))/5, 1);
                lw  = max(0.5, w*4);
                col = [0.2, 0.4+w*0.4, 1.0];
                line([nx2(i) nx2(j)],[ny2(i) ny2(j)],'Color',col,'LineWidth',lw);
            end
        end
    end
    node_colors = [0.3 0.7 1.0; 1.0 0.5 0.2; 0.3 0.9 0.5; 0.9 0.3 0.6; 0.9 0.8 0.2];
    node_colors = repmat(node_colors, ceil(n_rois/5), 1);
    for i = 1:n_rois
        scatter(nx2(i), ny2(i), 220, node_colors(i,:), 'filled', ...
                'MarkerEdgeColor','w','LineWidth',1.5);
        text(nx2(i)*1.28, ny2(i)*1.28, roi_labels{i}, ...
             'HorizontalAlignment','center','Color','w','FontSize',8,'FontWeight','bold');
    end
    axis equal off;
    title('Network graph','Color','w','FontWeight','bold');
    set(gca,'Color','k');
    sgtitle('Functional connectivity - NeuREI Lab | BINF 7550', ...
            'Color','w','FontSize',13,'FontWeight','bold');
    saveas(fig3, fullfile(OUT_DIR,'04_connectivity.png'));
    fprintf('    Saved: 04_connectivity.png\n');
end

% ---- Figure 4: Animated GIF ---------------------------------------------
fprintf('    Creating brain animation (~30s)...\n');
gif_path = fullfile(OUT_DIR, 'brain_activity_animation.gif');
fig4 = figure('Name','Animation','Position',[100 100 1000 420],'Color','k');

for z_anim = 1:nz
    clf;
    anat_sl2 = mean_anat(:,:,z_anim)';
    t_sl3    = t_ov(:,:,z_anim)';
    roi_sl2  = (roi_mask(:,:,z_anim) > 0)';

    ax1 = subplot(1,3,1);
    imshow(anat_sl2,[0 1],'Parent',ax1);
    colormap(ax1,gray);
    hold(ax1,'on');
    B2 = bwboundaries(roi_sl2);
    for b = 1:length(B2)
        plot(ax1, B2{b}(:,2), B2{b}(:,1), 'c-','LineWidth',1.8);
    end
    title(ax1,'Anatomy + ROIs','Color','w','FontSize',10,'FontWeight','bold');
    set(ax1,'Color','k');

    ax2 = subplot(1,3,2);
    safe_max = max(t_max_val, 0.01);
    imagesc(ax2, t_sl3, [0 safe_max]);
    axis(ax2,'image','off');
    colormap(ax2, hot);
    cb2 = colorbar(ax2); cb2.Color = 'w';
    title(ax2,'t-map','Color','w','FontSize',10,'FontWeight','bold');
    set(ax2,'Color','k');

    ax3 = subplot(1,3,3);
    [~,pv] = max(abs(t_sl3(:)));
    [px,py] = ind2sub([nx,ny], pv);
    ts   = squeeze(data4D(py, px, z_anim, :));
    ts_z = (ts - mean(ts)) / (std(ts) + eps);
    t_scan = (0:nt-1)*TR;
    area(ax3, t_scan, ts_z,'FaceColor',[0.2 0.6 1],'EdgeColor','none','FaceAlpha',0.7);
    xlabel(ax3,'Time (s)','Color','w');
    ylabel(ax3,'BOLD (z)','Color','w');
    ylim(ax3,[-4 4]); xlim(ax3,[0 max(t_scan)]);
    title(ax3,'Peak voxel BOLD','Color','w','FontSize',10,'FontWeight','bold');
    set(ax3,'Color','k','XColor','w','YColor','w');

    sgtitle(sprintf('Slice %d/%d - NeuREI Lab fMRI Pipeline', z_anim, nz), ...
            'Color','w','FontSize',11,'FontWeight','bold');
    drawnow;

    frame = getframe(fig4);
    img   = frame2im(frame);
    [imind, cm] = rgb2ind(img, 256);
    if z_anim == 1
        imwrite(imind, cm, gif_path,'gif','Loopcount',inf,'DelayTime',0.1);
    else
        imwrite(imind, cm, gif_path,'gif','WriteMode','append','DelayTime',0.1);
    end
end
close(fig4);
fprintf('    Saved: brain_activity_animation.gif\n');

fprintf('\n=== Pipeline Complete ===\n');
fprintf('All outputs saved to: %s\n', OUT_DIR);
fprintf('Finished: %s\n', datestr(now));