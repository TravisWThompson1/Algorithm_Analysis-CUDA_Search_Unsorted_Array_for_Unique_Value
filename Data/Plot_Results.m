% Read in data sets.
strided = textread('Strided_Offset_N_Search.txt');
coalesced = textread('Coalesced_N_Search.txt');
unrolled = textread('Unrolled_Coalesced_N_Search.txt');
full = textread('Full_Coalesced_Search.txt');
full(:,1) = coalesced(:,1);


% Create figure.
figure1 = figure('Name','Computation Time Comparision');
% Create axes.
axes1 = axes('Parent',figure1);
hold(axes1,'on');
% Plot data.
plot(strided(:,1),   strided(:,2),   'Linewidth', 2.0, 'Color', [0 0.447 0.741]);
plot(coalesced(:,1), coalesced(:,2), 'Linewidth', 2.0, 'Color', [0.85 0.325 0.098]);
plot(unrolled(:,1),  unrolled(:,2),  'Linewidth', 2.0), 'Color', [1 0 1];
plot(full(:,1),      full(:,2),      'Linewidth', 2.0), 'Color', [0.470588235294118 0.670588235294118 0.188235294117647];    
legend('Strided', 'Coalesced', 'Unrolled Coalesced', 'Full Coalesced');
% Create labels
ylabel('Computational Runtime [ms]');
xlabel('Inquiries per thread [N]');
% Set axis parameters.
xlim(axes1,[0 64]);
ylim(axes1,[0.01 0.05]);
box(axes1,'on');
grid(axes1,'on');
% Set the remaining axes properties.
set(axes1,'FontSize',16,'XTick',[0 8 16 24 32 40 48 56 64],'YTick',...
    [0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.045 0.05]);
% Create legend.
legend1 = legend(axes1,'show');
set(legend1,'Position',[0.627948679543435 0.773515742131372 0.25831281216861 0.136859942775175],...
    'FontSize',10);



