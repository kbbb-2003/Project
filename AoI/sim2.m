delta_t = 5;
lambda_t = 10;  % arg1
N1 = 100;
mu = 1 / (lambda_t);
a = exprnd(mu, N1, 1);
beta = 2;  % arg2
b = zeros(N1, 1);
for i = 1: N1-1
    b(i+1) = b(i) + a(i+1);
    if b(i+1) > delta_t
        N = i;
    end
end
b = b(1: N);
a = a(1: N);
z = zeros(10, 1);

figure(3)
ax = gca;
ax.LineWidth = 1;

for i = 1: N-1
    hold on
    aoi_n1 = a(i);
    aoi_n2 = a(i) + a(i+1);
    x = linspace(b(i), b(i+1), 10);
    y = linspace(aoi_n1, aoi_n2, 10);
    gain = exp(-beta*y);
    plot(x, gain, 'b')
    plot([b(i+1), b(i+1)], [0, exp(-beta*aoi_n2)], 'b-');
    fill([x, fliplr(x)], [gain, 0*ones(1, length(gain))], 'c')
    if i ~= N-1
        plot([b(i+1), b(i+1)], [exp(-beta*aoi_n2), exp(-beta*a(i+1))], 'b--');
    end
end

% G_r^t
k = zeros(N-1, 1);
plot(b(2: N, 1), k, 'r*');
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$\mathcal{G}_r^t$', 'Interpreter', 'latex');
legend('$\lambda_r=10, \ \beta_r=2$', 'Interpreter', 'latex');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 24);
axis([0, delta_t, 0, 1.2]);
