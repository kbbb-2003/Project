delta_t = 5;

% minus exp dis 1 (blue curve 1)
lambda_t = 1;   % lambda_1
N1 = 100;
mu = 1 / (lambda_t);
a = exprnd(mu, N1, 1);
beta = 1;   % beta_1
b = zeros(N1, 1);
for i = 1: N1 - 1
    b(i+1) = b(i) + a(i+1);
    if b(i+1) > delta_t
        N = i;
    end
end
b = b(1: N);
a = a(1: N);

% minus exp dis 2 (red curve 2)
lambda_t = 2;   % lambda_2
N1 = 100;
mu = 1 / (lambda_t);
a2 = exprnd(mu, N1, 1);
beta2 = 5;  % beta_2
b2 = zeros(N1, 1);
for i = 1: N1 - 1
    b2(i+1) = b2(i) + a2(i+1);
    if b2(i+1) > delta_t
        N = i;
    end
end
b2 = b2(1: N);
a2 = a2(1: N);

T = ['\it\fontname{Times New Roman}t_1';
     '\it\fontname{Times New Roman}t_2';
     '\it\fontname{Times New Roman}t_3';
     '\it\fontname{Times New Roman}t_4';
     '\it\fontname{Times New Roman}t_5'];

current = 1;
figure(1)
legend('off')
ax = gca;
ax.LineWidth = 1;

% the blue curve (curve 1)
for i = 1: N-1
    hold on
    aoi_n1 = a(i);
    aoi_n2 = a(i) + a(i+1);
    x = linspace(b(i), b(i+1), 10);
    y = linspace(aoi_n1, aoi_n2, 10);
    gain = exp(-beta*y);
    if i == 1
        % plot(x, gain, 'b', 'DisplayName', '\lambda_{\it\fontname{Times New Roman}p}=, \beta_{\it\fontname{Times New Roman}p}=')
        h(current) = plot(x, gain, 'b', 'LineWidth', 1);
        current = current + 1;
    else
        h(current) = plot(x, gain, 'b', 'LineWidth', 1);
        set(h(current), 'handlevisibility', 'off');
        current = current + 1;
    end
    h(current) = plot([b(i+1), b(i+1)], [0, exp(-beta*aoi_n2)], 'b-', 'LineWidth', 1);
    set(h(current), 'handlevisibility', 'off');
    current = current + 1;
    if (b(i) + b(i+1)) / 2 < delta_t && i <= 5
        text((b(i) + b(i+1)) / 2, 0.07, T(i, :), 'FontSize', 24);
    end
    if i ~= N-1
        h(current) = plot([b(i+1), b(i+1)], [exp(-beta*aoi_n2), exp(-beta*a(i+1))], 'b--', 'LineWidth', 1);
        set(h(current), 'handlevisibility', 'off');
        current = current + 1;
    end
end

k = zeros(N-1, 1);
h(current) = plot(b(2: N, 1), k, 'b', 'LineWidth', 1);
set(h(current), 'handlevisibility', 'off');
current = current + 1;

% the red curve (curve 2)
for i = 1: N-1
    hold on
    aoi_n1 = a2(i);
    aoi_n2 = a2(i) + a2(i+1);
    x = linspace(b2(i), b2(i+1), 10);
    y = linspace(aoi_n1, aoi_n2, 10);
    gain = exp(-beta2*y);
    if i == 1
        % plot(x, gain, 'r', 'DisplayName', '\lambda_{\it\fontname{Times New Roman}p}=, \beta_{\it\fontname{Times New Roman}p}=')
        h(current) = plot(x, gain, 'r', 'LineWidth', 1);
        current = current + 1;
    else
        h(current) = plot(x, gain, 'r', 'LineWidth', 1);
        set(h(current) ,'handlevisibility', 'off');
        current = current + 1;
    end
    h(current) = plot([b2(i+1), b2(i+1)], [0, exp(-beta2*aoi_n2)], 'r-', 'LineWidth', 1);
    set(h(current), 'handlevisibility', 'off');
    current = current + 1;
    if (b2(i) + b2(i+1)) / 2 < delta_t && i <= 5
        text((b2(i) + b2(i+1)) / 2, 0.07, T(i, :), 'FontSize', 24);   
    end
    if i ~= N-1
        h(current) = plot([b2(i+1), b2(i+1)], [exp(-beta2*aoi_n2), exp(-beta2*a2(i+1))], 'r--', 'LineWidth', 1);
        set(h(current), 'handlevisibility', 'off');
        current = current + 1;
    end
end

% g_r^t
k = zeros(N-1, 1);
plot(b2(2: N, 1), k, 'r', 'LineWidth', 1);
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$g_r^t$', 'Interpreter', 'latex');
legend('$\lambda_r=1, \ \beta_r=1$', '$\lambda_r=2, \ \beta_r=5$', 'Interpreter', 'latex');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 24);
axis([0, delta_t, 0, 1]);

current = 1
figure(2)
ax = gca;
ax.LineWidth = 1;

% the blue curve (curve 1)
for i = 1: N-1
    hold on
    aoi_n1 = a(i);
    aoi_n2 = a(i) + a(i+1);
    x = linspace(b(i), b(i+1), 10);
    y = linspace(aoi_n1, aoi_n2, 10);
    % gain = exp(-beta*y);
    if i == 1
        h(current) = plot(x, y, 'b', 'LineWidth', 1);
        current = current + 1;
    else
        h(current) = plot(x, y, 'b', 'LineWidth', 1);
        set(h(current), 'handlevisibility', 'off');
        current = current +1;
    end
    h(current) = plot([b(i), b(i+1)], [0, a(i+1)], 'b--', 'LineWidth', 1);
    set(h(current), 'handlevisibility', 'off');
    current = current + 1;
    h(current) = plot([b(i+1), b(i+1)], [a(i+1), aoi_n2], 'b-', 'LineWidth', 1);
    set(h(current), 'handlevisibility', 'off');
    current = current + 1;
    if (b(i)+b(i+1))/2 < delta_t && i <= 5
        text((b(i) + b(i+1)) / 2, 0.2, T(i, :), 'FontSize', 24);
    end
    if i ~= N-1
         h(current) = plot([b(i+1), b(i+1)], [0, a(i+1)], 'r--', 'LineWidth', 1);
         set(h(current), 'handlevisibility', 'off');
         current = current + 1;
     end
end
k = zeros(N-1, 1);
h(current) = plot(b(2: N, 1), k, 'b', 'LineWidth', 1);
set(h(current), 'handlevisibility', 'off');
current = current + 1;

% the red curve (curve 2)
for i = 1: N-1
    hold on
    aoi_n1 = a2(i);
    aoi_n2 = a2(i) + a2(i+1);
    x = linspace(b2(i), b2(i+1), 10);
    y = linspace(aoi_n1, aoi_n2, 10);
    % gain = exp(-beta*y);
    if i == 1
        h(current) = plot(x, y, 'r', 'LineWidth', 1);
        current = current + 1;
    else
        h(current) = plot(x, y, 'r', 'LineWidth', 1);
        set(h(current), 'handlevisibility', 'off');
        current = current + 1;
    end
    h(current) = plot([b2(i), b2(i+1)], [0, a2(i+1)], 'r--', 'LineWidth', 1);
    set(h(current), 'handlevisibility', 'off');
    current = current + 1;
    h(current) = plot([b2(i+1), b2(i+1)], [a2(i+1), aoi_n2], 'r-', 'LineWidth', 1);
    set(h(current), 'handlevisibility', 'off');
    current = current + 1;
    if (b2(i) + b2(i+1)) / 2 < delta_t && i <= 5
        text((b2(i) + b2(i+1)) / 2, 0.2, T(i, :), 'FontSize', 24);
    end
    if i ~= N-1
         h(current) = plot([b2(i+1), b2(i+1)], [0, a2(i+1)], 'r--', 'LineWidth', 1);
         set(h(current), 'handlevisibility', 'off');
         current = current + 1;
     end
end

% A_r^t
k = zeros(N-1, 1);
h(current) = plot(b2(2: N, 1), k, 'r', 'LineWidth', 1);
set(h(current), 'handlevisibility', 'off');
current = current + 1;
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$A_r^t$', 'Interpreter', 'latex');
legend('$\lambda_r=1, \ \beta_r=1$', '$\lambda_r=2, \ \beta_r=5$', 'Interpreter', 'latex');
axis([0, delta_t, 0, 3.5]);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 24);
