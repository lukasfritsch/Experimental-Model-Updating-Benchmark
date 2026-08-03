# return mean and standard deviation from DataFrame of samples
function get_mean_std(df::DataFrame)
    cols = [:m1, :m2, :m3, :m4, :k1, :k2, :k3, :k4]

    μ = [mean(col) for col in eachcol(df[:, cols])]
    σ = [std(col) for col in eachcol(df[:, cols])]

    return μ, σ
end

# plot mean and standard deviation
function plot_summary_statistics(results)
    labels_theta = [latexstring("\\theta_$i") for i in 1:8]

    colors = Makie.wong_colors()

    fig = Figure(size=(600, 400))

    axes = (
        mean_br=Axis(fig[1, 1], title=L"Data set $\biD^{(\mathrm{BR})}$", ylabel="Posterior Mean", titlefont=:regular),
        mean_de=Axis(fig[1, 2], title=L"Data set $\biD^{(\mathrm{DE})}$", titlefont=:regular),
        std_br=Axis(fig[2, 1], title="", ylabel="Posterior Std.", xticklabelsize=16,),
        std_de=Axis(fig[2, 2], title="", xticklabelsize=16),
    )

    for ax in values(axes)
        ax.xticks = (1:8, labels_theta)
    end

    nmethods = length(results)
    barwidth = 0.8 / nmethods

    legendplots = Makie.Plot[]

    for (i, res) in enumerate(results)
        x = (1:8) .+ (i - (nmethods + 1)/2) * barwidth

        μ_de, σ_de = res.de
        μ_br, σ_br = res.br

        μ_de = Float64.(coalesce.(μ_de, NaN))
        μ_br = Float64.(coalesce.(μ_br, NaN))
        σ_de = Float64.(coalesce.(σ_de, NaN))
        σ_br = Float64.(coalesce.(σ_br, NaN))

        p = barplot!(
            axes.mean_de, x, μ_de;
            width=barwidth,
            color=colors[i]
        )
        push!(legendplots, p)

        barplot!(axes.mean_br, x, μ_br;
            width=barwidth,
            color=colors[i]
        )

        barplot!(axes.std_de, x, σ_de;
            width=barwidth,
            color=colors[i]
        )

        barplot!(axes.std_br, x, σ_br;
            width=barwidth,
            color=colors[i]
        )
    end

    Legend(
        fig[0, 1:2],
        legendplots,
        [r.name for r in results];
        orientation=:horizontal,
    )

    rowgap!(fig.layout, 10)
    colgap!(fig.layout, 15)

    linkyaxes!(axes.mean_br, axes.mean_de)
    ylims!(axes.mean_br, 0.8, 1.21)
    linkyaxes!(axes.std_br, axes.std_de)

    # Right column: keep ticks, hide only labels
    hideydecorations!(
        axes.mean_de;
        label=true,
        ticklabels=true,
        ticks=false,
        grid=false,
        minorgrid=false,
        minorticks=false,
    )

    hideydecorations!(
        axes.std_de;
        label=true,
        ticklabels=true,
        ticks=false,
        grid=false,
        minorgrid=false,
        minorticks=false,
    )

    # Top row: keep ticks, hide only labels
    hidexdecorations!(
        axes.mean_br;
        label=true,
        ticklabels=true,
        ticks=false,
        grid=false,
        minorgrid=false,
        minorticks=false,
    )

    hidexdecorations!(
        axes.mean_de;
        label=true,
        ticklabels=true,
        ticks=false,
        grid=false,
        minorgrid=false,
        minorticks=false,
    )

    return fig
end
